"""
test_wiring.py -- smoke test del collegamento algo <-> venue SENZA rete e SENZA chiavi.
Una venue finta riempie gli ordini IOC con un po' di slippage; il bot gira su dati sintetici.
Verifica: formattazione prezzi/size, equivalenza dei pesi live, ciclo ingresso/uscita, stato su disco,
coerenza tra posizione del bot e posizione "sull'exchange", gestione di buchi nelle barre.

Uso:  python test_wiring.py
"""
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hl_connect import Venue, coin, fmt_price, fmt_size
from live_bot import Bot, LiveCfg, latest_weights
from pairs_pipeline_v3 import Cfg, CostModel, prepare, simulate_prices_v3
from train_model import fit_final


class FakeVenue(Venue):
    def __init__(self, prices):
        self.prices, self.i, self.dry_run, self.addr, self.exchange = prices, 0, False, "0x0", object()
        self.sz_dec = {coin("BTC"): 4, coin("ETH"): 4, coin("HYPE"): 2}
        self.max_lev = {c: 10 for c in self.sz_dec}
        self.held, self.orders = {}, []
        self.book_usd, self.book_step_bps = 50_000.0, 2.0       # liquidita' per livello e distanza tra livelli

    def l2_levels(self, c):
        px = float(self.prices.iloc[self.i][c.split(":")[1]])
        mk = lambda sgn: [dict(px=px * (1 + sgn * (k + 1) * self.book_step_bps / 1e4), sz=self.book_usd / px)
                          for k in range(5)]
        return mk(-1), mk(+1)

    def oracle_prices(self, assets):
        return self.prices.iloc[self.i][list(assets)]

    def set_leverage(self, c, lev):
        return None

    def positions(self):
        return {c: s for c, s in self.held.items() if abs(s) > 1e-12}

    def account_value(self):
        return 1_000_000.0

    def ioc_batch(self, legs, slip_bps, reduce_only=False):
        res = []
        for l in legs:
            sz = fmt_size(l["sz"], self.sz_dec[l["coin"]])
            px = l["ref_px"] * (1 + (1 if l["is_buy"] else -1) * slip_bps / 2e4)
            self.held[l["coin"]] = self.held.get(l["coin"], 0.0) + (sz if l["is_buy"] else -sz)
            self.orders.append((l["coin"], l["is_buy"], sz, reduce_only))
            res.append(dict(filled=sz, avg_px=px, error=None))
        return res


def test_formats():
    assert fmt_price(112345.67, 4) == 112350.0
    assert fmt_price(2696.437, 4) == 2696.4
    assert fmt_price(91.95123, 2) == 91.951
    assert fmt_size(0.123456, 4) == 0.1234 and fmt_size(1.239, 2) == 1.23
    print("formati prezzo/size OK")


def test_latest_weights(prices, cfg):
    prep = prepare(prices, cfg)
    w_roll = prep["W"].iloc[-1].values                       # finestra [t-W, t-1], applicata alla barra t
    w_live = latest_weights(prep["rets"].iloc[:-1], cfg)
    assert np.allclose(w_roll, w_live), (w_roll, w_live)
    assert abs(np.abs(w_live).sum() - 1) < 1e-9
    print("pesi live == pesi rolling OK:", w_live.round(3))


def test_bot(prices, cfg, cm, steps=36):
    tmp = tempfile.mkdtemp()
    bundle = fit_final(prices.iloc[:N0], cfg, cm, verbose=False)
    bundle["threshold"] = 0.0                                 # forza gli ingressi: qui si testa il cablaggio
    lc = LiveCfg(dry_run=False, max_gross_usd=2000.0, hist_window=2000, state_dir=f"{tmp}/state", log_dir=f"{tmp}/logs")
    venue = FakeVenue(prices)
    bot = Bot(venue, bundle, cfg, cm, lc, prices.iloc[:N0].copy())
    kinds = []
    for i in range(N0, N0 + steps):
        venue.i = i
        out = bot.step(now=prices.index[i] + pd.Timedelta(hours=1))
        kinds.append(out.get("action") or ("hold" if out.get("in_position") else "none"))
        # invariante: posizione del bot <=> posizione sull'exchange finto
        assert (bot.pos is not None) == bool(venue.positions()), (i, out, venue.positions())
    print("azioni:", pd.Series(kinds).value_counts().to_dict(), "| ordini:", len(venue.orders))
    assert (Path(tmp) / "logs").exists() and any((Path(tmp) / "logs").iterdir())
    if bot.pos:
        gross = bot.pos["gross"]
        assert gross <= lc.max_gross_usd * 1.02, gross
        assert all(abs(u) >= 10 for u in bot.pos["pos_usd"]), bot.pos["pos_usd"]
        print("posizione aperta, gross $%.0f, gambe $%s" % (gross, np.round(bot.pos["pos_usd"], 0).tolist()))
    # buco nelle barre: il bot non deve aprire posizioni
    lc2 = replace(lc, state_dir=f"{tmp}/state2", log_dir=f"{tmp}/logs2")      # stato separato
    bot2 = Bot(FakeVenue(prices), bundle, cfg, cm, lc2, prices.iloc[:N0].drop(prices.index[N0 - 100:N0 - 98]).copy())
    bot2.venue.i = N0
    out = bot2.step(now=prices.index[N0] + pd.Timedelta(hours=1))
    assert out["contiguous"] is False and out.get("action") in (None, "none", "wait_bars"), out
    print("buco nelle barre -> nessun ingresso OK")
    shutil.rmtree(tmp)


def test_liquidity_gate(prices, cfg, cm):
    """Book profondo -> size piena; medio -> size ridotta; sottile -> nessun ingresso."""
    tmp = tempfile.mkdtemp()
    bundle = fit_final(prices.iloc[:N0], cfg, cm, verbose=False)
    bundle["threshold"] = 0.0
    res = {}
    for name, depth, step in (("deep", 50_000.0, 0.5), ("medium", 800.0, 25.0), ("thin", 5.0, 3.0)):
        lc = LiveCfg(dry_run=False, max_gross_usd=2000.0, hist_window=2000,
                     state_dir=f"{tmp}/s_{name}", log_dir=f"{tmp}/l_{name}", max_cost_bps=20.0)
        venue = FakeVenue(prices)
        venue.book_usd, venue.book_step_bps = depth, step
        bot = Bot(venue, bundle, cfg, cm, lc, prices.iloc[:N0].copy())
        acts, gross = [], []
        for i in range(N0, N0 + 40):
            venue.i = i
            out = bot.step(now=prices.index[i] + pd.Timedelta(hours=1))
            acts.append(out.get("action") or "")
            if out.get("action") == "ENTER":
                gross.append(bot.pos["gross"])
                venue.held.clear(); bot.pos = None                 # chiude a mano: si prova solo l'ingresso
        res[name] = (pd.Series(acts).value_counts().to_dict(), gross)
        print(f"  book {name:6s}: {res[name][0]}  gross entrati: {np.round(gross, 0).tolist()[:4]}")
    assert res["deep"][1] and max(res["deep"][1]) > 1500, res["deep"]
    assert "liq_skip" in res["thin"][0] and not res["thin"][1], res["thin"]
    if res["medium"][1]:
        assert max(res["medium"][1]) < max(res["deep"][1]), res["medium"]
    shutil.rmtree(tmp)
    print("filtro di liquidita' OK")


def test_regime():
    from pairs_pipeline_v3 import entry_mask, size_scale, z_threshold
    c = Cfg(dyn_z=True, dyn_size=True, skip_vol_ratio=2.5)
    vr = np.array([0.3, 0.7, 1.0, 1.5, 2.0, 4.0])
    thr, sc = z_threshold(c, vr), size_scale(c, vr)
    assert np.all(np.diff(thr) >= 0) and np.all(np.diff(sc) <= 0)             # piu' vol -> soglia su, size giu'
    assert thr[0] >= c.z_entry * c.z_mult_lo - 1e-12 and thr[-1] <= c.z_entry * c.z_mult_hi + 1e-12
    assert sc.min() >= c.size_min and sc.max() <= c.size_max
    F = pd.DataFrame({"z": [-2.0] * 6, "vol_ratio": vr})
    assert entry_mask(F, c).tolist() == [True, True, True, True, True, False]  # 4.0 > skip
    off = Cfg()
    assert np.allclose(z_threshold(off, vr), off.z_entry) and np.allclose(size_scale(off, vr), 1.0)
    print("soglia/size dinamiche OK: thr", thr.round(2).tolist(), "size", sc.round(2).tolist())


def test_ewma(prices, cfg):
    ce = replace(cfg, ewma=True)
    prep = prepare(prices, ce)
    w_live = latest_weights(prep["rets"].iloc[:-1], ce)
    assert np.allclose(prep["W"].iloc[-1].values, w_live) and abs(np.abs(w_live).sum() - 1) < 1e-9
    base = prepare(prices, cfg)
    assert not np.allclose(prep["W"].iloc[-1].values, base["W"].iloc[-1].values)
    assert prep["F"]["z"].iloc[-1] != base["F"]["z"].iloc[-1]
    print("EWMA: pesi live == backtest OK:", w_live.round(3))


N0 = 2600
if __name__ == "__main__":
    cfg = replace(Cfg(), garch_min_train=900, garch_train_win=1500)
    cm = CostModel()
    prices = simulate_prices_v3(n=N0 + 60, mean_revert=True, seed=5, half_life=8, ou_innov=0.010)
    test_formats()
    test_latest_weights(prices.iloc[:1500], cfg)
    test_bot(prices, cfg, cm)
    test_liquidity_gate(prices, cfg, cm)
    test_regime()
    test_ewma(prices.iloc[:1500], cfg)
    print("TUTTI I TEST OK")
    