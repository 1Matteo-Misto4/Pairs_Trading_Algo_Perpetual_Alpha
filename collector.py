"""
collector.py -- logger dei dati di mercato, da lasciare acceso 24/7. NON richiede chiavi (query pubbliche).

Cosa registra (cartella data/collector):
  mkt_YYYYMMDD.csv        ogni --every secondi, per ogni asset del dex `par`:
        oracolo, mark, mid, funding orario (+ annualizzato), premium, open interest (monete e $), volume 24h,
        best bid/ask e size, profondita' entro 5/10/25 bps (bid e ask), COSTO REALE di eseguire a mercato
        $2k/$10k/$50k (slippage vs oracolo, camminando il book), differenze mark/mid/book vs oracolo,
        + riferimento MAINNET (oracolo, mid, funding, premium) per gli asset che esistono anche li'
  funding_history.csv     storico funding orario del dex `par` (testnet), aggiornato ogni ora
  funding_history_mainnet.csv   storico funding orario REALE di mainnet (backfill di --days giorni all'avvio):
        serve subito per testare una strategia di carry, senza aspettare di accumulare dati
  account_YYYYMMDD.csv    (solo con --team) equity e posizioni del team: base per rendimenti e deviazione giornalieri
  book_YYYYMMDD.jsonl     ogni --book-every secondi, i primi 10 livelli del book di ogni asset (analisi dell'impatto)
  limits.jsonl            ogni ora, il payload grezzo di perpDexLimits (cap di open interest ecc.)

Peso richieste per ciclo: ctx 20 + l2Book 2 x asset (+ mainnet ctx 20). Limite IP: 1200/min.

Uso:
    python collector.py --assets BTC ETH HYPE SP500 EWY EWJ BRENT GOLD USDEUR USDJPY --every 60
    python collector.py --assets ALL --every 60        # tutti i mercati del dex
    python collector.py --once                         # un solo snapshot a video (test), poi esce
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from hl_connect import Venue, coin

log = logging.getLogger("collector")
MAINNET = "https://api.hyperliquid.xyz/info"
DEFAULT_ASSETS = ["BTC", "ETH", "HYPE", "SP500", "EWY", "EWJ", "BRENT", "GOLD", "USDEUR", "USDJPY"]
BANDS_BPS = (5, 10, 25)
NOTIONALS = (2_000, 10_000, 50_000)
FOLDER = Path("data/collector")


# ---------------------------------------------------------------------- utilita'
def post_mainnet(payload, timeout=15):
    r = requests.post(MAINNET, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def depth_within(levels, ref, band_bps, side):
    lim = ref * (1 - band_bps / 1e4) if side == "bid" else ref * (1 + band_bps / 1e4)
    return sum(float(l["px"]) * float(l["sz"]) for l in levels
               if (float(l["px"]) >= lim if side == "bid" else float(l["px"]) <= lim))


def walk_cost_bps(levels, notional, oracle, side):
    """Slippage in bps contro l'ORACOLO per eseguire `notional` $ a mercato. NaN se il book non basta."""
    rem, spent, qty = float(notional), 0.0, 0.0
    for l in levels:
        px, sz = float(l["px"]), float(l["sz"])
        take = min(rem, px * sz)
        spent += take
        qty += take / px
        rem -= take
        if rem <= 1e-9:
            break
    if rem > 1e-6 or qty == 0:
        return np.nan
    avg = spent / qty
    return (avg - oracle) / oracle * 1e4 if side == "buy" else (oracle - avg) / oracle * 1e4


def mainnet_reference():
    """{asset: dict(ref_oracle, ref_mid, ref_funding, ref_premium)} dai mercati perp standard di mainnet."""
    meta, ctx = post_mainnet({"type": "metaAndAssetCtxs"})
    return {u["name"]: dict(ref_oracle=num(c.get("oraclePx")), ref_mid=num(c.get("midPx")),
                            ref_funding=num(c.get("funding")), ref_premium=num(c.get("premium")))
            for u, c in zip(meta["universe"], ctx)}


# ---------------------------------------------------------------------- snapshot
def snapshot(venue, assets, ts, ref=None, books=None):
    ctx = venue.ctxs()
    names = list(ctx.index) if assets == ["ALL"] else [coin(a) for a in assets]
    rows = []
    for c in names:
        try:
            r = ctx.loc[c]
            orc, mark, mid = num(r["oraclePx"]), num(r["markPx"]), num(r.get("midPx"))
            fund = num(r["funding"])
            imp = r.get("impactPxs")
            row = dict(ts=ts, coin=c, oracle=orc, mark=mark, mid=mid, funding=fund,
                       funding_ann_pct=fund * 24 * 365 * 100, premium=num(r.get("premium")),
                       oi_coins=num(r["openInterest"]), oi_usd=num(r["openInterest"]) * orc,
                       day_vol_usd=num(r.get("dayNtlVlm")), prev_day_px=num(r.get("prevDayPx")),
                       impact_bid_px=num(imp[0]) if isinstance(imp, (list, tuple)) and len(imp) == 2 else np.nan,
                       impact_ask_px=num(imp[1]) if isinstance(imp, (list, tuple)) and len(imp) == 2 else np.nan,
                       mark_vs_oracle_bps=(mark - orc) / orc * 1e4, mid_vs_oracle_bps=(mid - orc) / orc * 1e4)
            bids, asks = venue._post({"type": "l2Book", "coin": c})["levels"]
            if bids and asks:
                bb, ba = float(bids[0]["px"]), float(asks[0]["px"])
                bm = (bb + ba) / 2
                row.update(bid=bb, ask=ba, bid_sz=float(bids[0]["sz"]), ask_sz=float(asks[0]["sz"]),
                           spread_bps=(ba - bb) / bm * 1e4,
                           bid_vs_oracle_bps=(orc - bb) / orc * 1e4, ask_vs_oracle_bps=(ba - orc) / orc * 1e4)
                for b in BANDS_BPS:
                    row[f"depth_bid_{b}bps_usd"] = depth_within(bids, bm, b, "bid")
                    row[f"depth_ask_{b}bps_usd"] = depth_within(asks, bm, b, "ask")
                for n in NOTIONALS:
                    row[f"cost_buy_{n}_bps"] = walk_cost_bps(asks, n, orc, "buy")
                    row[f"cost_sell_{n}_bps"] = walk_cost_bps(bids, n, orc, "sell")
                # nomi compatibili con liquidity_report (profondita' entro 10 bps)
                row["depth_bid_usd"], row["depth_ask_usd"] = row["depth_bid_10bps_usd"], row["depth_ask_10bps_usd"]
                if books is not None:
                    books.append(dict(ts=str(ts), coin=c, bids=[(l["px"], l["sz"]) for l in bids[:10]],
                                      asks=[(l["px"], l["sz"]) for l in asks[:10]]))
            else:
                row.update(bid=np.nan, ask=np.nan, depth_bid_usd=0.0, depth_ask_usd=0.0)
            a = c.split(":", 1)[-1]
            if ref and a in ref:
                rf = ref[a]
                row.update(rf)
                row["oracle_vs_ref_bps"] = (orc - rf["ref_oracle"]) / rf["ref_oracle"] * 1e4
                row["mid_vs_ref_bps"] = (mid - rf["ref_mid"]) / rf["ref_mid"] * 1e4
                row["funding_vs_ref"] = fund - rf["ref_funding"]
            rows.append(row)
        except Exception:
            log.exception("snapshot %s fallito (gli altri asset proseguono)", c)
    return rows


def append_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():                                            # allinea le colonne se l'insieme cambia
        old_cols = list(pd.read_csv(path, nrows=0).columns)
        for col in new.columns.difference(old_cols):
            log.warning("colonna nuova %s non presente in %s: ignorata per oggi", col, path.name)
        new = new.reindex(columns=old_cols)
    new.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------------- funding storico
def last_times(path):
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return df.groupby("coin")["time_ms"].max().to_dict() if len(df) else {}


def fetch_funding(post, coin_name, query_coin, start_ms, max_pages=40):
    out, t = [], int(start_ms)
    for _ in range(max_pages):
        rows = post({"type": "fundingHistory", "coin": query_coin, "startTime": t})
        if not rows:
            break
        out += [dict(coin=coin_name, time_ms=int(x["time"]), time=pd.to_datetime(int(x["time"]), unit="ms", utc=True),
                     funding=float(x["fundingRate"]), premium=num(x.get("premium"))) for x in rows]
        if len(rows) < 500:
            break
        t = int(rows[-1]["time"]) + 1
    return out


def update_funding_history(venue, names, days, mainnet=True):
    now_ms = int(time.time() * 1000)
    for label, path, post, items in (
        ("par", FOLDER / "funding_history.csv", venue._post, [(c, c) for c in names]),
        ("mainnet", FOLDER / "funding_history_mainnet.csv", post_mainnet,
         [(c.split(":", 1)[-1],) * 2 for c in names] if mainnet else []),
    ):
        last = last_times(path)
        for cn, qc in items:
            try:
                start = last.get(cn, now_ms - days * 86_400_000) + 1
                rows = fetch_funding(post, cn, qc, start)
                if rows:
                    append_csv(rows, path)
                    log.info("funding %s %s: +%d righe", label, cn, len(rows))
            except Exception as e:
                log.info("funding %s %s non disponibile (%s)", label, cn, type(e).__name__)


# ---------------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, help="nomi senza prefisso, oppure ALL")
    ap.add_argument("--every", type=int, default=60, help="secondi tra due snapshot")
    ap.add_argument("--team", default=None, help="indirizzo del team: se dato registra anche equity e posizioni")
    ap.add_argument("--days", type=int, default=90, help="giorni di funding storico da scaricare all'avvio")
    ap.add_argument("--book-every", type=int, default=600, help="secondi tra due dump dei primi 10 livelli (0=mai)")
    ap.add_argument("--no-mainnet", action="store_true", help="non interrogare mainnet (riferimento e funding reale)")
    ap.add_argument("--once", action="store_true", help="un solo snapshot a video, poi esce")
    a = ap.parse_args()
    if a.team and not re.fullmatch(r"0x[0-9a-fA-F]{40}", a.team):
        raise SystemExit(f"--team non e' un indirizzo valido (0x + 40 caratteri esadecimali): {a.team!r}")

    venue = Venue(a.team or "0x0000000000000000000000000000000000000000", api_key=None, dry_run=True)
    names = [coin(x) for x in a.assets] if a.assets != ["ALL"] else list(venue.ctxs().index)
    FOLDER.mkdir(parents=True, exist_ok=True)

    if a.once:
        ref = None if a.no_mainnet else mainnet_reference()
        rows = snapshot(venue, a.assets, pd.Timestamp.now(tz="UTC"), ref)
        with pd.option_context("display.width", 250, "display.max_columns", 60):
            print(pd.DataFrame(rows).set_index("coin").T.to_string())
        return

    try:
        lim = venue.perp_limits()
        log.info("perpDexLimits: %s", lim)
        with open(FOLDER / "limits.jsonl", "a") as f:
            f.write(json.dumps(dict(ts=str(pd.Timestamp.now(tz="UTC")), limits=lim), default=str) + "\n")
    except Exception:
        log.exception("perpDexLimits non letto")
    update_funding_history(venue, names, a.days, mainnet=not a.no_mainnet)

    t_fund = t_book = t_lim = time.time()
    while True:
        t0 = time.time()
        ts = pd.Timestamp.now(tz="UTC")
        try:
            ref = None
            if not a.no_mainnet:
                try:
                    ref = mainnet_reference()
                except Exception as e:
                    log.warning("riferimento mainnet non raggiunto (%s): snapshot senza", type(e).__name__)
            want_book = a.book_every > 0 and time.time() - t_book >= a.book_every
            books = [] if want_book else None
            rows = snapshot(venue, a.assets, ts, ref, books)
            if rows:
                append_csv(rows, FOLDER / f"mkt_{ts:%Y%m%d}.csv")
            if want_book and books:
                with open(FOLDER / f"book_{ts:%Y%m%d}.jsonl", "a") as f:
                    f.writelines(json.dumps(b) + "\n" for b in books)
                t_book = time.time()
            if a.team:
                try:                                                 # un errore qui NON deve bloccare il resto del ciclo
                    st = venue._post({"type": "clearinghouseState", "user": a.team, "dex": "par"})
                    append_csv([dict(ts=ts, equity=venue.account_value(), positions=json.dumps(venue.positions()),
                                     margin_used=num(st.get("marginSummary", {}).get("totalMarginUsed")))],
                               FOLDER / f"account_{ts:%Y%m%d}.csv")
                except Exception as e:
                    log.warning("account non letto (%s: %s)", type(e).__name__, str(e)[:120])
            if time.time() - t_fund >= 3600:
                update_funding_history(venue, names, a.days, mainnet=not a.no_mainnet)
                t_fund = time.time()
            if time.time() - t_lim >= 3600:
                with open(FOLDER / "limits.jsonl", "a") as f:
                    f.write(json.dumps(dict(ts=str(ts), limits=venue.perp_limits()), default=str) + "\n")
                t_lim = time.time()
        except Exception:
            log.exception("ciclo fallito: si riprova al prossimo")
        time.sleep(max(0.0, a.every - (time.time() - t0)))


if __name__ == "__main__":
    main()
