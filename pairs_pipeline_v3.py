"""
pairs_pipeline_v3.py -- pipeline orientata ai criteri di valutazione di Perpetual Alpha.

Criteri della competizione (dalle regole):
  (1) rendimento netto totale sul capitale iniziale, marcato all'ORACOLO a fine Live
  (2) -std dei rendimenti giornalieri dell'equity  (meno volatilita' = punteggio piu' alto)
  (3) giuria sul report  |  vincoli: >=10 giorni distinti di trading, volume minimo, entrate algoritmiche
  (1) e (2) sono normalizzati min-max tra i team (non osservabili: i valori "peers" sono PARAMETRI).

Cosa cambia rispetto a v2 (che resta il motore di segnale: PCA causale, GARCH causale, purged CV):
  * Universo BTC/ETH/HYPE (SOL non esiste su `par`).
  * Sizing in DOLLARI sul capitale: vol-target giornaliero, tetto leva, tetto OI per gamba,
    tetto profondita', arrotondamento al lot, minimo $10 per ordine.
  * Equity oraria marcata all'oracolo (serie di prezzi del backtest), poi equity GIORNALIERA UTC:
    la std e' quella dell'equity (posizioni aperte incluse), non dei soli PnL a chiusura.
  * Costi misurati contro l'oracolo: fee maker/taker + mezzo spread vs oracolo + impatto + funding orario.
  * Rischio di gamba: probabilita' di rifiuto della copertura (cap OI) con costo di unwind.
  * Governor di partecipazione (>=10 giorni distinti) e metriche di volume.
  * Punteggio proxy min-max e "scale hint": dice se conviene scalare la size verso i tetti.

I valori marcati PLACEHOLDER (spread, profondita', tetto OI, peers) vanno sostituiti con misure
della fase Practice e con i dati reali della leaderboard.
"""
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pairs_pipeline_v2 import (build_signal_and_features, make_model, purged_oof,
                               simulate_prices, uniqueness_weights)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- configurazione
@dataclass
class Cfg:
    # segnale (stessi campi usati dalle funzioni di v2)
    assets: tuple = ("BTC", "ETH", "HYPE")
    leg_a: str = "BTC"
    leg_b: str = "ETH"
    pca_win: int = 336
    z_win: int = 48
    z_entry: float = 1.2
    horizon: int = 24
    entry_delay: int = 1
    tp_mult: float = 0.6
    sl_mult: float = 0.4
    garch_refit: int = 168
    garch_min_train: int = 1500
    garch_train_win: int = 3000
    test_frac: float = 0.30
    cooldown_after_sl: int = 6
    n_splits: int = 5
    seed: int = 42
    # capitale e rischio
    capital: float = 100_000.0         # PLACEHOLDER: capitale iniziale del team
    vol_target: float = 0.005          # std giornaliera target dell'equity (frazione del capitale)
    max_leverage: float = 3.0          # tetto sul notional lordo / capitale
    oi_cap_leg_usd: float = 5_000.0    # PLACEHOLDER: quota di cap OI ($100k/mercato) utilizzabile per gamba
    depth_frac: float = 0.25           # al piu' questa frazione della profondita' visibile
    min_notional: float = 10.0
    max_gross_usd: float = 1e12        # tetto assoluto sul notional lordo (Practice: tenerlo basso)
    lots: dict = field(default_factory=lambda: {"BTC": 1e-4, "ETH": 1e-4, "HYPE": 1e-2})
    ref_gross: float = 5_000.0         # notional di riferimento per le label (costi con impatto)
    leg_fail_prob: float = 0.05        # P(rifiuto della copertura): cap OI, margine, errore API
    # partecipazione
    min_days: int = 10
    gov_buffer: int = 3
    relax_th: float = 0.0
    # EWMA (default OFF: i bundle gia' addestrati restano validi). Se attivo va RIADDESTRATO il modello.
    ewma: bool = False
    z_hl: float = 24.0                 # emivita (ore) di media/dev.std dello z-score
    pca_hl: float = 168.0              # emivita (ore) dei pesi nella PCA (finestra comunque = pca_win)
    # regime di volatilita' (usa vol_ratio = GARCH / mediana 168h, causale). Default OFF.
    dyn_z: bool = False                # soglia di ingresso z dinamica
    z_vol_alpha: float = 0.5           # soglia = z_entry * clip(vol_ratio**alpha, z_mult_lo, z_mult_hi)
    z_mult_lo: float = 0.85
    z_mult_hi: float = 1.6
    dyn_size: bool = False             # size dinamica: gross = base * clip(1/vol_ratio, size_min, size_max)
    size_min: float = 0.25
    size_max: float = 1.25
    skip_vol_ratio: float = 0.0        # se > 0: niente ingresso quando vol_ratio supera questo valore


# --------------------------------------------------------------------------- regime di volatilita'
def z_threshold(cfg: Cfg, vr):
    """Soglia |z| di ingresso: fissa, oppure z_entry scalata dalla volatilita' relativa (vettoriale)."""
    if not cfg.dyn_z:
        return cfg.z_entry + 0.0 * np.asarray(vr, dtype=float)
    m = np.clip(np.power(np.asarray(vr, dtype=float), cfg.z_vol_alpha), cfg.z_mult_lo, cfg.z_mult_hi)
    return cfg.z_entry * m


def size_scale(cfg: Cfg, vr):
    """Moltiplicatore della size: mercato calmo (vol_ratio<1) -> di piu', agitato -> di meno."""
    if not cfg.dyn_size:
        return 1.0 + 0.0 * np.asarray(vr, dtype=float)
    return np.clip(1.0 / np.asarray(vr, dtype=float), cfg.size_min, cfg.size_max)


def entry_mask(F: pd.DataFrame, cfg: Cfg) -> np.ndarray:
    """Barre in cui scatta un ingresso: feature complete, |z| >= soglia (dinamica) e regime non troppo agitato."""
    vr = F["vol_ratio"].values
    ok = F.notna().all(axis=1).values & (F["z"].abs().values >= z_threshold(cfg, vr))
    if cfg.skip_vol_ratio > 0:
        ok &= vr <= cfg.skip_vol_ratio
    return ok


@dataclass
class CostModel:
    fee_maker_bps: float = 3.0
    fee_taker_bps: float = 9.0
    # PLACEHOLDER: mezzo spread del book locale vs ORACOLO, in bps (da misurare in Practice)
    half_spread_bps: tuple = (1.0, 1.5, 4.0)
    depth_usd: tuple = (20_000.0, 20_000.0, 10_000.0)   # PLACEHOLDER: profondita' entro pochi bps
    impact_bps_at_depth: float = 2.0                    # bps extra se il notional = profondita'
    entry_maker_frac: float = 0.5                       # quota di notional entrata come maker
    tp_maker: bool = True                               # TP come ordine limit maker

    def side_cost_usd(self, n_abs, maker_frac):
        sp, dp = np.array(self.half_spread_bps), np.array(self.depth_usd)
        fee = maker_frac * self.fee_maker_bps + (1 - maker_frac) * self.fee_taker_bps
        taker_extra = (1 - maker_frac) * (sp + self.impact_bps_at_depth * n_abs / dp)
        return float((n_abs * (fee + taker_extra)).sum() / 1e4)

    def round_trip_frac(self, absw, gross, kind):
        n = absw * gross
        exit_m = 1.0 if (kind == "tp" and self.tp_maker) else 0.0
        return (self.side_cost_usd(n, self.entry_maker_frac) + self.side_cost_usd(n, exit_m)) / gross


# --------------------------------------------------------------------------- eventi e barriere
def barrier_events(event_pos, Pm, rets, W, vol, z, cfg: Cfg, cm: CostModel, vr=None) -> pd.DataFrame:
    A = rets[list(cfg.assets)].values
    vrv = np.ones(len(A)) if vr is None else np.asarray(vr, dtype=float)
    S = np.cumsum(A, axis=0)
    Wv, volv, zv = W.values, vol.values, z.values
    n, H = len(A), cfg.horizon
    rows = []
    for t in event_pos:
        e = t + cfg.entry_delay
        if e + H + 1 >= n:
            continue
        w, sig = Wv[e + 1], volv[e + 1]
        if np.isnan(w).any() or np.isnan(sig):
            continue
        d = -np.sign(zv[t])
        cum = S[e + 1:e + H + 1] - S[e]                       # rendimento cumulato per asset
        path = d * (cum @ w)                                  # PnL su notional lordo = 1
        tp, sl = cfg.tp_mult * sig * np.sqrt(H), cfg.sl_mult * sig * np.sqrt(H)
        i_tp, i_sl = np.flatnonzero(path >= tp), np.flatnonzero(path <= -sl)
        i_tp, i_sl = (i_tp[0] if len(i_tp) else H), (i_sl[0] if len(i_sl) else H)
        if i_tp == H and i_sl == H:
            kind, k = "time", H - 1
        elif i_tp < i_sl:
            kind, k = "tp", i_tp
        else:
            kind, k = "sl", i_sl
        gross = path[k]
        net_ref = gross - cm.round_trip_frac(np.abs(w), cfg.ref_gross, kind)
        rows.append(dict(t_pos=t, entry_pos=e, exit_pos=e + 1 + k, kind=kind, gross=gross,
                         net=net_ref, hold=k + 1, d=d, w=w, sig=sig, cum=cum[:k + 1], px=Pm[e], vr=vrv[t]))
    out = pd.DataFrame(rows)
    out["t0"] = Pm_index(rets, out["t_pos"])
    out["t1"] = Pm_index(rets, out["exit_pos"])
    out["y"] = (out["net"] > 0).astype(int)
    return out.set_index("t0")


def Pm_index(rets, pos):
    return rets.index[np.asarray(pos)]


# --------------------------------------------------------------------------- sizing e simulazione
MIN_LEG_W = 0.02   # gambe con |peso| sotto questa soglia (es. HYPE ~0.0007) non si tradano: notional 0


def size_trade(r, cfg: Cfg, cm: CostModel):
    """Notional firmato per gamba in $ (0 per le gambe trascurabili), oppure None se viola minimo/lot."""
    w = np.asarray(r.w, dtype=float)
    active = np.abs(w) >= MIN_LEG_W                                    # gambe davvero tradate
    if not active.any():
        return None
    absw, sgn = np.where(active, np.abs(w), 0.0), np.sign(w)
    G = cfg.vol_target * cfg.capital / (r.sig * np.sqrt(24))          # vol-target giornaliero
    G = min(G, cfg.max_leverage * cfg.capital, cfg.max_gross_usd)      # tetto leva e notional assoluto
    G *= float(size_scale(cfg, getattr(r, "vr", 1.0)))                 # regime: calmo -> piu' size, agitato -> meno
    caps = np.minimum(cfg.oi_cap_leg_usd, cfg.depth_frac * np.array(cm.depth_usd))
    G = min(G, float((caps[active] / absw[active]).min()))             # tetto OI/profondita' per gamba
    lots = np.array([cfg.lots[a] for a in cfg.assets])
    size = np.floor(G * absw / r.px / lots + 1e-9) * lots              # arrotonda per difetto al lot
    notional = size * r.px
    if (notional[active] < cfg.min_notional).any():                    # minimo $10 solo sulle gambe attive
        return None
    return sgn * notional


def run_sim(ev, probs, th, cfg: Cfg, cm: CostModel, idx, rates, start_pos, end_pos,
            governor=False, seed=0):
    rng = np.random.default_rng(seed)
    inc = np.zeros(len(idx))
    days_all = pd.date_range(idx[start_pos].floor("D"), idx[end_pos].floor("D"), freq="D")
    traded, volume, trades = set(), 0.0, []
    skipped = {"min_notional": 0, "leg_fail": 0}
    free_at = pd.Timestamp.min.tz_localize(idx.tz)
    for i, r in enumerate(ev.itertuples()):
        p = probs[i]
        if np.isnan(p) or r.Index < free_at or r.entry_pos < start_pos or r.exit_pos > end_pos:
            continue
        day, th_i = idx[r.entry_pos].floor("D"), th
        if governor and day not in traded:
            days_left = int((days_all >= day).sum())
            if cfg.min_days - len(traded) + cfg.gov_buffer >= days_left:
                th_i = min(th, cfg.relax_th)                           # rilassa la soglia per partecipare
        if p < th_i:
            continue
        n_s = size_trade(r, cfg, cm)
        if n_s is None:
            skipped["min_notional"] += 1
            continue
        n_abs = np.abs(n_s)
        traded.add(day)
        if rng.random() < cfg.leg_fail_prob:                           # copertura rifiutata -> unwind
            j = int(np.argmax(n_abs))
            big = np.zeros_like(n_abs)
            big[j] = n_abs[j]
            inc[r.entry_pos] -= 2 * cm.side_cost_usd(big, 0.0)
            volume += 2 * n_abs[j]
            skipped["leg_fail"] += 1
            continue
        pos = r.d * n_s
        pathd = r.d * (r.cum @ n_s)                                    # PnL $ cumulato, marcato all'oracolo
        hrs = r.entry_pos + 1 + np.arange(len(pathd))
        inc[hrs] += np.diff(np.r_[0.0, pathd])
        inc[hrs] -= rates[hrs] @ pos                                   # funding: chi e' long paga se rate>0
        exit_m = 1.0 if (r.kind == "tp" and cm.tp_maker) else 0.0
        inc[r.entry_pos] -= cm.side_cost_usd(n_abs, cm.entry_maker_frac)
        inc[r.exit_pos] -= cm.side_cost_usd(n_abs, exit_m)
        volume += 2 * n_abs.sum()
        traded.add(idx[r.exit_pos].floor("D"))
        free_at = r.t1 + (pd.Timedelta(hours=cfg.cooldown_after_sl) if r.kind == "sl" else pd.Timedelta(0))
        trades.append((r.Index, r.t1, r.kind, float(pathd[-1]), float(n_abs.sum())))
    m = competition_metrics(inc, idx, start_pos, end_pos, cfg, len(trades), traded, volume)
    m["skipped"] = skipped
    return m


# --------------------------------------------------------------------------- metriche competizione
def competition_metrics(inc, idx, start_pos, end_pos, cfg: Cfg, n_trades, traded, volume):
    seg = inc[start_pos:end_pos + 1]
    eq = pd.Series(cfg.capital + np.cumsum(seg), index=idx[start_pos:end_pos + 1])
    daily = eq.groupby(eq.index.floor("D")).last()
    dret = daily.pct_change()
    dret.iloc[0] = daily.iloc[0] / cfg.capital - 1
    R = float(eq.iloc[-1] / cfg.capital - 1)
    sig = float(dret.std(ddof=1))
    d = dret.values
    rng, bs = np.random.default_rng(0), []
    for _ in range(1000):                                              # block bootstrap (blocchi di 5 giorni)
        ix = np.concatenate([(rng.integers(0, len(d)) + np.arange(5)) % len(d)
                             for _ in range(int(np.ceil(len(d) / 5)))])[:len(d)]
        bs.append(d[ix].mean() / (d[ix].std() + 1e-12) * np.sqrt(365))
    return {"R": R, "sigma_d": sig,
            "sharpe": float(d.mean() / (sig + 1e-12) * np.sqrt(365)),
            "sharpe_ci90": tuple(np.percentile(bs, [5, 95]).round(2)),
            "max_dd": float(((eq.cummax() - eq) / cfg.capital).max()),
            "n_trades": n_trades, "days_traded": len(traded), "volume_usd": float(volume),
            "days_ok": len(traded) >= cfg.min_days}


def competition_proxy(m, peers, w_r=1.0, w_s=1.0):
    """Punteggio min-max sui soli criteri quantitativi, dati i range PLACEHOLDER dei peers."""
    rn = np.clip((m["R"] - peers["R_min"]) / (peers["R_max"] - peers["R_min"]), 0, 1)
    sn = np.clip((peers["sig_max"] - m["sigma_d"]) / (peers["sig_max"] - peers["sig_min"]), 0, 1)
    return float((w_r * rn + w_s * sn) / (w_r + w_s))


def scale_hint(m, peers, w_r=1.0, w_s=1.0):
    """
    Se la size scala di k, R e sigma_d scalano di k: il punteggio e' lineare in k, quindi
    conviene salire verso i tetti se  w_r*R/dR > w_s*sigma/dSigma, altrimenti scendere.
    """
    dR, dS = peers["R_max"] - peers["R_min"], peers["sig_max"] - peers["sig_min"]
    gain, cost = w_r * m["R"] / dR, w_s * m["sigma_d"] / dS
    return {"gain_R": gain, "cost_sigma": cost, "scale_up": bool(gain > cost)}


# --------------------------------------------------------------------------- pipeline
def prepare(prices: pd.DataFrame, cfg: Cfg):
    """Parte pesante e indipendente da sizing/costi (PCA e GARCH causali): calcolala una volta."""
    rets = np.log(prices[list(cfg.assets)]).diff().dropna()
    W, pnl, vol, F = build_signal_and_features(rets, cfg)
    Pm = prices[list(cfg.assets)].reindex(rets.index).values
    return dict(rets=rets, W=W, vol=vol, F=F, Pm=Pm)


def evaluate(prep, cfg: Cfg, cm: CostModel, peers, gate=None, funding=None, verbose=True):
    rets, W, vol, F, Pm = prep["rets"], prep["W"], prep["vol"], prep["F"], prep["Pm"]
    idx = rets.index
    rates = (funding.reindex(idx).fillna(0.0)[list(cfg.assets)].values if funding is not None
             else np.zeros((len(idx), len(cfg.assets))))

    ok = entry_mask(F, cfg)
    if gate is not None:                                              # gate esterno (VIX, DVOL, vol realizzata)
        ok &= gate.reindex(idx).fillna(True).astype(bool).values
    out = barrier_events(np.flatnonzero(ok), Pm, rets, W, vol, F["z"], cfg, cm, vr=F["vol_ratio"].values)
    X, y = F.loc[out.index], out["y"]

    t_split = out.index[int(len(out) * (1 - cfg.test_frac))]
    tr = (out["t1"] < t_split).values
    te = out.index >= t_split + pd.Timedelta(hours=cfg.horizon)
    Xtr, ytr, otr, Xte, ote = X[tr], y[tr], out[tr], X[te], out[te]
    wtr = uniqueness_weights(otr, len(idx))

    # soglia scelta su OOF del train, con l'equity simulata (Sharpe giornaliero), min 20 trade
    oof = purged_oof(Xtr, ytr, otr.index.to_series(), otr["t1"], wtr, cfg)
    s0, e0 = int(otr["t_pos"].min()), int(otr["exit_pos"].max())
    best_th, best_sh = 1.1, -np.inf
    for th in np.linspace(0.30, 0.70, 41):
        m = run_sim(otr, oof, th, cfg, cm, idx, rates, s0, e0)
        if m["n_trades"] >= 20 and m["sharpe"] > best_sh:
            best_th, best_sh = th, m["sharpe"]

    model = make_model(cfg.seed).fit(Xtr, ytr, sample_weight=wtr)
    p_te = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(ote["y"], p_te) if ote["y"].nunique() > 1 else np.nan

    s1, e1 = int(ote["t_pos"].min()), len(idx) - 1                    # finestra calendario fino a fine dati
    base = run_sim(ote, np.ones(len(ote)), 0.0, cfg, cm, idx, rates, s1, e1, governor=True)
    meta = run_sim(ote, p_te, best_th, cfg, cm, idx, rates, s1, e1, governor=True)
    res = dict(threshold=best_th, auc=auc, n_events=len(out), base=base, meta=meta,
               proxy_base=competition_proxy(base, peers), proxy_meta=competition_proxy(meta, peers),
               hint=scale_hint(meta, peers),
               _ctx=dict(ote=ote, p_te=p_te, idx=idx, rates=rates, s1=s1, e1=e1, th=best_th))
    if verbose:
        print(f"Eventi {len(out)} (train {len(otr)}, test {len(ote)}) | soglia {best_th:.2f} | AUC test {auc:.3f}")
        for k in ("base", "meta"):
            m = res[k]
            print(f"  {k:4s} R={m['R']*100:7.3f}%  sigma_d={m['sigma_d']*100:6.3f}%  Sharpe={m['sharpe']:6.2f} "
                  f"CI90={m['sharpe_ci90']}  maxDD={m['max_dd']*100:5.2f}%  trade={m['n_trades']:3d}  "
                  f"giorni={m['days_traded']:2d}{'' if m['days_ok'] else ' (<MIN!)'}  "
                  f"vol=${m['volume_usd']:,.0f}  proxy={res['proxy_'+k]:.2f}  skip={m['skipped']}")
        h = res["hint"]
        print(f"  scale hint: guadagno marginale {h['gain_R']:.3f} vs costo {h['cost_sigma']:.3f} -> "
              f"{'SCALARE verso i tetti' if h['scale_up'] else 'RIDURRE / non scalare'}")
    return res


def sweep(res, cfg: Cfg, cm: CostModel, peers, param, values):
    """Frontiera (R, sigma) al variare di un parametro di rischio, sul test (solo per il report)."""
    c = res["_ctx"]
    print(f"  sweep {param}:")
    for v in values:
        m = run_sim(c["ote"], c["p_te"], c["th"], replace(cfg, **{param: v}), cm, c["idx"], c["rates"],
                    c["s1"], c["e1"], governor=True)
        print(f"    {param}={v:<9} R={m['R']*100:7.3f}%  sigma_d={m['sigma_d']*100:6.3f}%  "
              f"R/sigma={m['R']/(m['sigma_d']+1e-12):7.2f}  proxy={competition_proxy(m, peers):.2f}")


# --------------------------------------------------------------------------- dati sintetici (test)
def simulate_prices_v3(n=9000, mean_revert=False, seed=0, **kw):
    px = simulate_prices(n=n, mean_revert=mean_revert, seed=seed, **kw).rename(columns={"SOL": "HYPE"})
    return px * np.array([1000.0, 40.0, 0.4])        # livelli realistici: ~100k, ~4k, ~40


PEERS = dict(R_min=-0.02, R_max=0.05, sig_min=0.0005, sig_max=0.03)   # PLACEHOLDER: da leaderboard

if __name__ == "__main__":
    cfg, cm = Cfg(), CostModel()
    for name, kw in (("PLACEBO (random walk)", dict(mean_revert=False, seed=1)),
                     ("CONTROLLO POSITIVO (OU forte)", dict(mean_revert=True, seed=3, half_life=8, ou_innov=0.010))):
        print(f"\n=== {name} ===")
        prep = prepare(simulate_prices_v3(**kw), cfg)
        res = evaluate(prep, cfg, cm, PEERS)
        if "POSITIVO" in name:
            sweep(res, cfg, cm, PEERS, "vol_target", [0.001, 0.003, 0.01, 0.03])
            sweep(res, cfg, cm, PEERS, "oi_cap_leg_usd", [1_000.0, 5_000.0, 20_000.0, 50_000.0])
