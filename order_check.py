"""
order_check.py -- prova che gli ordini FIRMATI funzionano, con rischio minimo (testnet, ~$15).

Cosa verifica, in ordine:
  A. ordine ALO (post-only) lontano dal mercato -> deve risultare `resting`, poi viene annullato
  B. andata e ritorno IOC su una gamba: compra ~$15, poi vende reduce-only fino a tornare flat
  C. misura reale di: slippage contro l'ORACOLO, fee pagata in USDPR (in bps), saldo prima/dopo
  D. rate limit dell'account

SICUREZZA
  * senza `--go` NON invia nulla: stampa solo il piano
  * rifiuta di partire se hai gia' una posizione sul coin, e limita il notional a $50
  * usa l'API wallet (.secrets/trader.key), mai la chiave del team wallet

Uso:
    python order_check.py --team 0x...                 # piano, nessun ordine
    python order_check.py --team 0x... --go            # esegue il test su par:ETH
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from hl_connect import DEX, Venue, coin, fmt_price, fmt_size, parse_statuses

MAX_NOTIONAL = 50.0


def check(ok, msg):
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)
    return ok


def test_resting_and_cancel(v: Venue, c: str, notional: float) -> bool:
    print("\nA. ordine ALO lontano dal mercato, poi cancellazione")
    orc = float(v.ctxs().loc[c, "oraclePx"])
    sd = v.sz_dec[c]
    px = fmt_price(orc * 0.90, sd)                                 # -10%: non incrocia, dentro la banda dell'80%
    sz = fmt_size(notional / px, sd)
    res = v.exchange.order(c, True, sz, px, {"limit": {"tif": "Alo"}}, False)
    st = parse_statuses(res, 1)[0]
    print("  risposta:", res)
    resting = "oid" in st
    check(resting, f"ordine in book (oid={st.get('oid')}) a {px} con size {sz}")
    if not resting:
        return False
    cres = v.exchange.cancel(c, st["oid"])
    print("  cancel:", cres)
    time.sleep(1)
    opens = v._post({"type": "openOrders", "user": v.addr, "dex": DEX})
    left = [o for o in opens if o.get("oid") == st["oid"]]
    return check(not left, "ordine non piu' presente tra gli aperti")


def test_ioc_round_trip(v: Venue, c: str, notional: float, slip_bps: float) -> bool:
    print("\nB/C. andata e ritorno IOC con misura di slippage e fee")
    bal0 = v.account_value()
    ctx = v.ctxs().loc[c]
    orc, sd = float(ctx["oraclePx"]), v.sz_dec[c]
    sz = fmt_size(notional / orc, sd)
    legs = [dict(coin=c, is_buy=True, sz=sz, ref_px=orc)]
    r_in = v.ioc_batch(legs, slip_bps, reduce_only=False)[0]
    print("  ingresso:", r_in)
    if r_in["error"] or r_in["filled"] <= 0:
        return check(False, "ingresso non eseguito (book vuoto o fuori dal limite di prezzo?)")
    slip_in = (r_in["avg_px"] / orc - 1) * 1e4
    check(True, f"ingresso riempito {r_in['filled']} @ {r_in['avg_px']}  (slippage vs oracolo {slip_in:+.1f} bps)")
    time.sleep(1)
    held = v.positions().get(c, 0.0)
    check(abs(held - r_in["filled"]) < 1e-9, f"posizione letta dall'exchange = {held}")

    ok_out, slip = False, slip_bps
    for k in range(3):                                             # uscita con limite via via piu' largo
        held = v.positions().get(c, 0.0)
        if abs(held) < 1e-12:
            ok_out = True
            break
        orc2 = float(v.ctxs().loc[c, "oraclePx"])
        r_out = v.ioc_batch([dict(coin=c, is_buy=held < 0, sz=abs(held), ref_px=orc2)], slip, reduce_only=True)[0]
        print(f"  uscita tentativo {k}:", r_out)
        slip *= 2
        time.sleep(1)
    ok_out = ok_out or abs(v.positions().get(c, 0.0)) < 1e-12
    check(ok_out, "posizione tornata a zero")

    fills = v._post({"type": "userFills", "user": v.addr})
    mine = [f for f in fills if f.get("coin") in (c, c.split(":")[-1])][:4]
    for f in mine:
        bps = float(f["fee"]) / (float(f["px"]) * float(f["sz"])) * 1e4
        print(f"  fill {f['side']} {f['sz']} @ {f['px']}  fee {f['fee']} {f.get('feeToken')}  = {bps:.2f} bps"
              f"  ({'taker' if f.get('crossed') else 'maker'})")
    if not mine:
        print("  (nessun fill in userFills: controlla se serve il parametro dex per questa query)")
    time.sleep(1)
    print(f"  saldo prima {bal0:,.4f} -> dopo {v.account_value():,.4f} USDPR")
    return ok_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=os.environ.get("TEAM_ADDRESS"))
    ap.add_argument("--key-file", default=os.environ.get("TRADER_KEY_FILE", ".secrets/trader.key"))
    ap.add_argument("--coin", default="ETH")
    ap.add_argument("--notional", type=float, default=15.0)
    ap.add_argument("--slip-bps", type=float, default=20.0)
    ap.add_argument("--go", action="store_true", help="invia davvero gli ordini")
    a = ap.parse_args()
    assert 10.0 <= a.notional <= MAX_NOTIONAL, f"notional tra $10 (minimo venue) e ${MAX_NOTIONAL}"
    c = coin(a.coin)

    if not a.go:
        print(f"PIANO (nessun ordine inviato): ALO {c} ~${a.notional} a -10% oracolo + cancel; "
              f"poi IOC buy ~${a.notional} con slippage max {a.slip_bps} bps e vendita reduce-only. Aggiungi --go.")
        return
    v = Venue(a.team, api_key=Path(a.key_file).read_text().strip(), dry_run=False)
    print("Saldo:", v.account_value(), "| posizioni:", v.positions())
    assert c not in v.positions(), f"gia' una posizione su {c}: chiudila prima"
    print("Leva:", v.set_leverage(c, 2))
    results = [test_resting_and_cancel(v, c, a.notional), test_ioc_round_trip(v, c, a.notional, a.slip_bps)]
    print("\nD. rate limit:", v.user_rate_limit())
    print("\nESITO:", "TUTTO OK" if all(results) else "CI SONO FALLIMENTI: incollami l'output")


if __name__ == "__main__":
    main()
