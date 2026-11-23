"""
live_bot.py -- collega l'algoritmo (v2/v3) alla venue. Un ciclo all'ora, a hh:00:10 UTC.

Ciclo `step()`:
  1. legge l'oracolo (`oraclePx`) di BTC/ETH/HYPE e aggiunge una barra oraria. Convenzione: il prezzo
     letto a hh:00 e' la CHIUSURA della candela con apertura (hh-1):00, come nei CSV storici.
  2. ricalcola pesi PCA, z-score, GARCH causale e feature con `prepare()` sulle ultime N barre
  3. se in posizione: controlla TP / SL / time-out sul PnL marcato all'oracolo -> uscita
  4. se flat e |z| >= soglia: probabilita' del modello; se >= soglia -> sizing -> ingresso
  5. guardia di rischio giornaliera, log JSON, stato su disco (sopravvive ai riavvii)

SICURO DI DEFAULT: senza `--live` NON invia ordini (dry-run: li simula e li logga).
Il segnale e' sull'ORACOLO, l'esecuzione sul book locale con ordini IOC e limite a ref +/- slippage.

Uso:
    python live_bot.py --team 0x... --model models/bundle.joblib --data-dir data/organizers            # dry-run
    python live_bot.py --team 0x... --model models/bundle.joblib --data-dir data/organizers --live
"""
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from hl_connect import Venue, coin, fmt_size, walk_cost_bps
from pairs_pipeline_v2 import pca_weights_window
from pairs_pipeline_v3 import Cfg, CostModel, prepare, size_scale, size_trade, z_threshold

log = logging.getLogger("bot")


@dataclass
class LiveCfg:
    dry_run: bool = True
    max_gross_usd: float = 2_000.0     # Practice: size minuscola (NON e' la size del Live)
    slip_entry_bps: float = 15.0       # limite di prezzo rispetto all'oracolo in ingresso (order_check: +10.9 bps su ETH)
    slip_exit_bps: float = 30.0        # in uscita: piu' largo, uscire conta piu' che risparmiare
    leverage: int = 3
    hist_window: int = 4000
    run_offset_s: int = 10
    daily_loss_frac: float = 0.005     # stop giornaliero: -0.5% dell'equity
    fill_tol: float = 0.98             # una gamba e' "piena" se riempita almeno al 98%
    liq_filter: bool = True            # prima di entrare simula l'esecuzione sul book reale
    max_cost_bps: float = 20.0         # costo atteso per lato (media pesata sulle gambe, vs oracolo): oltre -> riduce/salta
    liq_steps: tuple = (1.0, 0.75, 0.5, 0.35, 0.25)   # frazioni della size provate se il book e' sottile
    state_dir: str = "state"
    log_dir: str = "logs"


def latest_weights(rets: pd.DataFrame, cfg: Cfg) -> np.ndarray:
    """Stessa formula di rolling_pca_weights, per la PROSSIMA barra (finestra = ultime pca_win ore)."""
    return pca_weights_window(rets[list(cfg.assets)].values[-cfg.pca_win:], cfg)


class Bot:
    def __init__(self, venue, bundle, cfg: Cfg, cm: CostModel, lc: LiveCfg, bars: pd.DataFrame):
        self.venue, self.bundle, self.cfg, self.cm, self.lc = venue, bundle, cfg, cm, lc
        self.bars = bars
        Path(lc.state_dir).mkdir(exist_ok=True)
        Path(lc.log_dir).mkdir(exist_ok=True)
        self.state_path = Path(lc.state_dir) / "position.json"
        self.pos = json.loads(self.state_path.read_text()) if self.state_path.exists() else None
        self.cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        self.day, self.day_equity, self.halted = None, None, False
        self.coins = [coin(a) for a in cfg.assets]

    # ------------------------------------------------------------------ utilita'
    def _event(self, now, kind, **kw):
        rec = dict(ts=str(now), kind=kind, dry_run=self.lc.dry_run, **kw)
        with open(Path(self.lc.log_dir) / f"bot_{now:%Y%m%d}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log.info("%s %s", kind, kw)

    def _save(self):
        if self.pos:
            self.state_path.write_text(json.dumps(self.pos))
        elif self.state_path.exists():
            self.state_path.unlink()

    def _retry(self, fn, *args, tries=12, wait=30.0):
        """Ritenta una chiamata di SOLA LETTURA (rete caduta, DNS...). Mai usarla attorno agli ordini."""
        for k in range(tries):
            try:
                return fn(*args)
            except Exception as e:
                if k == tries - 1:
                    raise
                log.warning("rete/API non raggiungibile (%s): tentativo %d/%d, riprovo tra %.0fs",
                            type(e).__name__, k + 1, tries, wait)
                time.sleep(wait)

    def _heal_gaps(self):
        """Colma con le candele di mainnet le ore mancanti nelle ultime barre (es. dopo un'interruzione di rete)."""
        need = self.cfg.pca_win + self.cfg.z_win + 1
        tail = self.bars.index[-need:]
        if len(tail) < 2:
            return
        full = pd.date_range(tail[0], tail[-1], freq="h")
        missing = full.difference(tail)
        if len(missing) == 0:
            return
        try:
            from data_loader import backfill_mainnet
            extra = backfill_mainnet(self.cfg.assets, missing[0] - pd.Timedelta(hours=1), missing[-1] + pd.Timedelta(hours=1))
            extra = extra[extra.index.isin(missing)]
            if len(extra):
                self.bars = pd.concat([self.bars, extra[list(self.cfg.assets)]]).sort_index()
                self.bars = self.bars[~self.bars.index.duplicated(keep="last")]
            log.warning("barre mancanti %d, recuperate da mainnet %d", len(missing), len(extra))
        except Exception as e:
            log.warning("recupero barre mancanti fallito (%s): riprovo al prossimo ciclo", e)

    def _snapshot_bar(self, now):
        px = self._retry(self.venue.oracle_prices, self.cfg.assets)
        label = now.floor("h") - pd.Timedelta(hours=1)           # apertura della candela appena chiusa
        if label not in self.bars.index:
            self.bars.loc[label] = px[list(self.cfg.assets)].values
            self.bars = self.bars.sort_index()
        return label, px[list(self.cfg.assets)].values.astype(float)

    # ------------------------------------------------------------------ ciclo orario
    def step(self, now=None) -> dict:
        now = now or pd.Timestamp.now(tz="UTC")
        label, px_now = self._snapshot_bar(now)
        self._heal_gaps()
        bars = self.bars.iloc[-self.lc.hist_window:]
        need = self.cfg.pca_win + self.cfg.z_win + 1                 # barre che alimentano PCA e z-score
        tail = bars.index[-need:]
        contiguous = bool(len(tail) == need and (np.diff(tail.values) == np.timedelta64(1, 'h')).all())
        prep = prepare(bars, self.cfg)
        rets, vol, F = prep["rets"], prep["vol"], prep["F"]
        z, feats = float(F["z"].iloc[-1]), F.iloc[-1]
        vr = float(feats["vol_ratio"])
        out = {"label": str(label), "z": round(z, 3), "contiguous": contiguous}
        if np.isfinite(vr):                                          # regime di volatilita' sempre nel log
            out.update(vr=round(vr, 2), z_thr=round(float(z_threshold(self.cfg, vr)), 2),
                       size_x=round(float(size_scale(self.cfg, vr)), 2))

        self._risk_guard(now, px_now)
        if self.halted:
            out["halted"] = True
        elif self.pos:
            out.update(self._manage_position(now, px_now))
        elif contiguous and now >= self.cooldown_until:
            out.update(self._maybe_enter(now, px_now, z, feats, rets, vol))
        elif now < self.cooldown_until:                              # perche' non si entra: lo scriviamo nel log
            out.update(action="cooldown", until=str(self.cooldown_until))
        else:
            if len(tail) < need:
                gap = f"solo {len(tail)}/{need} barre"
            else:
                k = int(np.flatnonzero(np.diff(tail.values) != np.timedelta64(1, 'h'))[0])
                gap = f"buco dopo {tail[k]} (poi {tail[k + 1]})"
            out.update(action="wait_bars", gap=gap)
        self._save()
        self._event(now, "step", **out)
        return out

    # ------------------------------------------------------------------ ingresso
    def _maybe_enter(self, now, px_now, z, feats, rets, vol):
        cfg, b = self.cfg, self.bundle
        if feats.isna().any():
            return {"action": "none"}
        vr = float(feats["vol_ratio"])
        if abs(z) < float(z_threshold(cfg, vr)):
            return {"action": "none"}
        if cfg.skip_vol_ratio > 0 and vr > cfg.skip_vol_ratio:
            return {"action": "skip", "why": f"mercato troppo agitato (vol_ratio {vr:.2f} > {cfg.skip_vol_ratio})"}
        p = float(b["model"].predict_proba(feats[b["features"]].astype(float).to_frame().T)[0, 1])
        if p < b["threshold"]:
            return {"action": "filtered", "p": round(p, 3)}
        sig = float(vol.iloc[-1])
        if not np.isfinite(sig):
            return {"action": "skip", "why": "vol nan"}
        w, d = latest_weights(rets, cfg), -np.sign(z)
        r = SimpleNamespace(w=w, sig=sig, px=px_now, vr=vr)
        n_s = size_trade(r, replace(cfg, max_gross_usd=self.lc.max_gross_usd), self.cm)
        if n_s is None:
            return {"action": "skip", "why": "min_notional/lot", "w": w.round(4).tolist()}
        liq = {}
        if self.lc.liq_filter:
            n_s, liq = self._liquidity_gate(r, n_s)
            if n_s is None:
                return {"action": "liq_skip", "p": round(p, 3), **liq}

        pos_usd = d * n_s                                           # notional firmato per gamba (long > 0)
        act = [i for i, pu in enumerate(pos_usd) if abs(pu) > 0]    # gambe trascurabili (es. HYPE): nessun ordine
        legs = [dict(coin=self.coins[i], is_buy=bool(pos_usd[i] > 0), sz=abs(pos_usd[i]) / px_now[i],
                     ref_px=float(px_now[i])) for i in act]
        for l in legs:
            self.venue.set_leverage(l["coin"], self.lc.leverage)
        res = self.venue.ioc_batch(legs, self.lc.slip_entry_bps, reduce_only=False)
        full = [r["error"] is None and r["filled"] >= self.lc.fill_tol * fmt_size(l["sz"], self.venue.sz_dec[l["coin"]])
                for r, l in zip(res, legs)]
        if not all(full):                                           # leg failure: smonta quanto e' stato eseguito
            self._unwind_partial(now, legs, res)
            self.cooldown_until = now + pd.Timedelta(hours=1)
            return {"action": "LEG_FAIL", "p": round(p, 3), "res": res}

        signed = [0.0] * len(self.coins)                            # allineato a self.coins; 0 per le gambe non tradate
        for i, l, r in zip(act, legs, res):
            signed[i] = (1 if l["is_buy"] else -1) * r["filled"] * (r["avg_px"] or l["ref_px"])
        H = cfg.horizon
        self.pos = dict(entry_time=str(now), d=float(d), coins=self.coins, pos_usd=signed,
                        px_entry=px_now.tolist(), gross=float(np.abs(signed).sum()), sig=sig, p=p, z=z,
                        tp=cfg.tp_mult * sig * np.sqrt(H), sl=cfg.sl_mult * sig * np.sqrt(H), horizon_h=H)
        return {"action": "ENTER", "p": round(p, 3), "pos_usd": np.round(signed, 1).tolist(), "vr": round(vr, 2),
                **({"liq": liq} if liq else {})}

    def _liquidity_gate(self, r, n_s):
        """
        Simula l'esecuzione IOC sul book reale: per ogni gamba attiva costo (bps vs oracolo) del notional.
        Riduce la size (lc.liq_steps) finche' il costo medio pesato <= max_cost_bps; se il book non copre una
        gamba o neanche la size minima passa -> (None, info). Restituisce (notional da usare, info per il log).
        """
        act = [i for i, v in enumerate(n_s) if abs(v) > 0]
        try:
            books = {i: self._retry(self.venue.l2_levels, self.coins[i], tries=3, wait=2.0) for i in act}
        except Exception as e:
            return None, {"why": f"book non leggibile ({type(e).__name__})"}
        base = self.lc.max_gross_usd
        best = {}
        for f in self.lc.liq_steps:
            ns = n_s if f == 1.0 else size_trade(r, replace(self.cfg, max_gross_usd=base * f), self.cm)
            if ns is None:
                break
            costs, tot = {}, 0.0
            for i in act:
                n = abs(ns[i])
                bids, asks = books[i]
                side = "buy" if ns[i] > 0 else "sell"
                ref, sl = float(r.px[i]), self.lc.slip_entry_bps / 1e4
                lev = ([l for l in asks if float(l["px"]) <= ref * (1 + sl)] if side == "buy"       # solo i livelli che
                       else [l for l in bids if float(l["px"]) >= ref * (1 - sl)])                  # il limite IOC raggiunge
                c = walk_cost_bps(lev, n, ref, side)
                costs[self.coins[i]] = None if not np.isfinite(c) else round(float(c), 1)
                tot += n
            best = dict(frac=f, cost_bps=costs, gross=round(float(np.abs(ns).sum()), 0))
            if any(v is None for v in costs.values()):
                continue                                           # il book non copre: prova una size piu' piccola
            avg = sum(abs(ns[i]) * costs[self.coins[i]] for i in act) / tot
            best["avg_bps"] = round(float(avg), 1)
            if avg <= self.lc.max_cost_bps:
                return ns, best
        best["why"] = f"book troppo sottile / costoso (limite {self.lc.max_cost_bps} bps per lato)"
        return None, best

    def _unwind_partial(self, now, legs, res):
        back = [dict(coin=l["coin"], is_buy=not l["is_buy"], sz=r["filled"], ref_px=l["ref_px"])
                for l, r in zip(legs, res) if r["filled"] > 0]
        if back:
            self._event(now, "unwind", res=self.venue.ioc_batch(back, self.lc.slip_exit_bps, reduce_only=True))

    # ------------------------------------------------------------------ gestione posizione e uscita
    def _manage_position(self, now, px_now):
        P = self.pos
        cum = np.log(px_now / np.array(P["px_entry"]))
        ret = float(np.dot(P["pos_usd"], cum) / P["gross"])         # PnL su notional lordo, marcato all'oracolo
        held = (now - pd.Timestamp(P["entry_time"])).total_seconds() / 3600
        kind = "tp" if ret >= P["tp"] else "sl" if ret <= -P["sl"] else "time" if held >= P["horizon_h"] else None
        info = {"in_position": True, "ret_bps": round(ret * 1e4, 1), "held_h": round(held, 1)}
        if kind is None:
            return info
        ok = self._exit(now, px_now)
        info.update(action="EXIT", exit_kind=kind, flat=ok)
        if ok:
            self.pos = None
            if kind == "sl":
                self.cooldown_until = now + pd.Timedelta(hours=self.cfg.cooldown_after_sl)
        return info

    def _exit(self, now, px_now, attempts=3) -> bool:
        slip = self.lc.slip_exit_bps
        for k in range(attempts):
            if self.lc.dry_run:
                return True
            held = self.venue.positions()
            legs = [dict(coin=c, is_buy=held[c] < 0, sz=abs(held[c]), ref_px=float(px))
                    for c, px in zip(self.coins, px_now) if c in held]
            if not legs:
                return True
            self._event(now, "exit_try", attempt=k, slip_bps=slip,
                        res=self.venue.ioc_batch(legs, slip, reduce_only=True))
            slip *= 2                                                # allarga il limite a ogni tentativo
            time.sleep(1.0)
        left = self.venue.positions()
        self._event(now, "EXIT_INCOMPLETE", left=left)
        return not any(c in left for c in self.coins)

    # ------------------------------------------------------------------ rischio
    def _risk_guard(self, now, px_now):
        day = now.floor("D")
        eq = self._retry(self.venue.account_value)
        if day != self.day:
            self.day, self.day_equity, self.halted = day, eq, False
            return
        if self.day_equity and (eq - self.day_equity) / self.day_equity <= -self.lc.daily_loss_frac and not self.halted:
            self.halted = True
            self._event(now, "DAILY_LOSS_HALT", equity=eq, start=self.day_equity)
            if self.pos and self._exit(now, px_now):
                self.pos = None

    def run_forever(self):
        while True:
            now = pd.Timestamp.now(tz="UTC")
            nxt = now.ceil("h") + pd.Timedelta(seconds=self.lc.run_offset_s)
            if nxt - now > pd.Timedelta(hours=1, seconds=self.lc.run_offset_s):
                nxt = now
            time.sleep(max(0.0, (nxt - now).total_seconds()))
            try:
                self.step()
            except Exception:
                log.exception("step fallito: il bot resta attivo, si riprova all'ora successiva")


def load_bars(data_dir, cfg: Cfg, state_dir="state") -> pd.DataFrame:
    """Storico degli organizzatori + barre live gia' salvate + colma il buco con mainnet (se possibile)."""
    from data_loader import backfill_mainnet, load_history
    bars = load_history(data_dir, cfg.assets)
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    if bars.index[-1] < end - pd.Timedelta(hours=1):
        try:
            extra = backfill_mainnet(cfg.assets, bars.index[-1], end)
            bars = pd.concat([bars, extra[~extra.index.isin(bars.index)]]).sort_index()
            print(f"Backfill mainnet fino a {bars.index[-1]}")
        except Exception as e:
            print("Backfill non riuscito:", e, "-> il bot non tradera' finche' le barre non sono contigue")
    return bars


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=os.environ.get("TEAM_ADDRESS"))
    ap.add_argument("--key-file", default=os.environ.get("TRADER_KEY_FILE", ".secrets/trader.key"))
    ap.add_argument("--model", default="models/bundle.joblib")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-gross", type=float, default=2000.0)
    ap.add_argument("--live", action="store_true", help="invia ordini veri (altrimenti dry-run)")
    ap.add_argument("--no-filter", action="store_true",
                    help="TEST: disattiva il filtro del modello (soglia prob = 0): entra a ogni |z| >= z_entry")
    ap.add_argument("--no-liq-filter", action="store_true", help="disattiva il filtro di liquidita' sul book")
    ap.add_argument("--max-cost-bps", type=float, default=20.0,
                    help="costo atteso massimo per lato (bps vs oracolo, media pesata); oltre: riduce la size o salta")
    ap.add_argument("--leg-cap-usd", type=float, default=None,
                    help="tetto di notional per gamba (default 5000). Alzalo per size grandi: il limite vero lo mette il filtro di liquidita'")
    ap.add_argument("--slip-entry-bps", type=float, default=15.0, help="limite di prezzo IOC in ingresso vs oracolo")
    ap.add_argument("--ewma", action="store_true", help="z-score e PCA con pesi esponenziali (ricorda: riaddestra il modello)")
    ap.add_argument("--dyn-z", action="store_true", help="soglia z dinamica: sale con la volatilita', scende se calmo")
    ap.add_argument("--dyn-size", action="store_true", help="size dinamica: piu' grande se calmo, piu' piccola se agitato")
    ap.add_argument("--skip-vol-ratio", type=float, default=0.0, help="non entrare se vol_ratio supera questo valore (es. 2.5; 0=off)")
    ap.add_argument("--size-min", type=float, default=None)
    ap.add_argument("--size-max", type=float, default=None)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    if a.no_filter:
        bundle["threshold"] = 0.0
        log.warning("*" * 70)
        log.warning("MODALITA' TEST --no-filter: il modello NON filtra gli ingressi. Non usare per la gara.")
        log.warning("*" * 70)
    cfg, cm = bundle["cfg"], CostModel()
    if a.ewma != getattr(cfg, "ewma", False) and a.ewma:
        if not a.no_filter:
            log.warning("--ewma: il modello e' stato addestrato su feature NON-EWMA (z, mom...): riaddestra con train_model.py --ewma")
        cfg = replace(cfg, ewma=True)
        over = dict(dyn_z=a.dyn_z or getattr(cfg, "dyn_z", False), dyn_size=a.dyn_size or getattr(cfg, "dyn_size", False),
                skip_vol_ratio=a.skip_vol_ratio or getattr(cfg, "skip_vol_ratio", 0.0),
                z_entry=1.5, tp_mult=0.6, sl_mult=0.8)
    if a.size_min is not None:
        over["size_min"] = a.size_min
    if a.size_max is not None:
        over["size_max"] = a.size_max
    if a.leg_cap_usd:
        over["oi_cap_leg_usd"] = a.leg_cap_usd
        cm = replace(cm, depth_usd=(a.leg_cap_usd / cfg.depth_frac,) * 3)      # i segnaposto non devono strozzare: decide il book
    cfg = replace(Cfg(), **{**cfg.__dict__, **over})           # riempie i campi nuovi se il bundle e' vecchio
    lc = LiveCfg(dry_run=not a.live, max_gross_usd=a.max_gross, liq_filter=not a.no_liq_filter, max_cost_bps=a.max_cost_bps,
                  slip_entry_bps=a.slip_entry_bps)
    key = Path(a.key_file).read_text().strip() if a.live else None
    venue = Venue(a.team, api_key=key, dry_run=lc.dry_run)
    log.info("perpDexLimits (payload grezzo): %s", venue.perp_limits())
    log.info("Modalita': %s | gross max $%s", "DRY-RUN" if lc.dry_run else "LIVE", lc.max_gross_usd)
    log.info("EWMA=%s | z dinamico=%s | size dinamica=%s (x%.2f..x%.2f) | skip se vol_ratio>%s | filtro liquidita'=%s (max %.0f bps/lato)",
             cfg.ewma, cfg.dyn_z, cfg.dyn_size, cfg.size_min, cfg.size_max, cfg.skip_vol_ratio or "off",
             lc.liq_filter, lc.max_cost_bps)
    log.info("Modello: z_entry=%s | soglia prob=%.2f | eventi train=%s | addestrato fino a %s",
             cfg.z_entry, bundle["threshold"], bundle.get("n_events"), bundle.get("trained_to"))
    Bot(venue, bundle, cfg, cm, lc, load_bars(a.data_dir, cfg)).run_forever()


if __name__ == "__main__":
    main()
