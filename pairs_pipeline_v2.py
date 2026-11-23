"""
pairs_pipeline_v2.py  --  versione corretta della pipeline di 01_Algo.ipynb

Correzioni principali rispetto al notebook originale
 1. Spread = PnL di un portafoglio REALE con pesi espliciti (rolling PCA a 3 asset,
    pesi stimati su [t-W, t-1]); il PC1 viene neutralizzato includendo anche SOL/HYPE.
 2. GARCH(1,1)-t causale: refit ogni 168h su sola finestra passata, poi recursione a
    parametri fissi; stessa serie su cui si applicano le barriere (niente mismatch).
 3. Barriere scalate con sqrt(orizzonte); uscita al valore REALE della barra (gap incluso).
 4. Costi parametrici (fee per lato, slippage, funding) sottratti dal PnL.
 5. Entrata ritardata di `entry_delay` barre rispetto al segnale (niente fill al close del segnale).
 6. Label coerente: netto > 0 dopo costi, per ogni tipo di uscita (TP/SL/time-out).
 7. Eventi sovrapposti: pesi di unicita' + purged K-fold con embargo -> probabilita' OOF.
 8. Soglia scelta SOLO su OOF del train, con simulazione sequenziale (una posizione per volta).
 9. Test toccato una volta sola; metriche giornaliere (Sharpe sqrt(365)) + block bootstrap.
10. Test placebo (random walk) e controllo positivo (mean reversion iniettata).

Unita': il PnL e' rendimento sul notional LORDO (somma |pesi| = 1). I costi in bps sono
per lato sul notional lordo (con gambe uguali = media delle fee delle due gambe).
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from arch import arch_model
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class Cfg:
    assets: tuple = ("BTC", "ETH", "SOL")
    leg_a: str = "BTC"
    leg_b: str = "ETH"
    pca_win: int = 336
    z_win: int = 48
    z_entry: float = 1.2
    horizon: int = 24
    entry_delay: int = 1          # barre tra segnale ed entrata
    tp_mult: float = 0.6          # in unita' di sigma_h * sqrt(horizon)
    sl_mult: float = 0.4
    entry_bps: float = 6.0        # per lato, su notional lordo (mix maker/taker)
    exit_tp_bps: float = 3.0      # TP come limit maker
    exit_sl_bps: float = 9.0      # SL / time-out taker
    slip_bps: float = 1.0         # per lato
    funding_bps_h: float = 0.0    # costo orario su notional lordo (da calibrare)
    garch_refit: int = 168
    garch_min_train: int = 1500
    garch_train_win: int = 3000
    test_frac: float = 0.30
    cooldown_after_sl: int = 6
    n_splits: int = 5
    seed: int = 42


# ----------------------------------------------------------------------------- dati
def load_yfinance(tickers: dict, period="730d", interval="1h") -> pd.DataFrame:
    """Scarica prezzi e verifica buchi. Per la produzione usare le candele della venue."""
    import yfinance as yf
    raw = yf.download(list(tickers.values()), period=period, interval=interval,
                      auto_adjust=True, progress=False)["Close"]
    px = raw.rename(columns={v: k for k, v in tickers.items()})[list(tickers)]
    px = px.dropna()
    gaps = px.index.to_series().diff().dropna()
    n_gaps = int((gaps > pd.Timedelta(hours=1)).sum())
    if n_gaps:
        print(f"ATTENZIONE: {n_gaps} buchi temporali; i rendimenti attraverso un buco sono multi-ora.")
    return px


# ------------------------------------------------------------ spread e pesi (causali)
def pca_weights_window(win: np.ndarray, cfg) -> np.ndarray:
    """
    Pesi (gross=1) del portafoglio residuo  e_a - e_b - (v_a - v_b) v  su UNA finestra di rendimenti
    (righe = ore, la piu' vecchia per prima). v = PC1 della correlazione.
    Se cfg.ewma: media/varianza/correlazione pesate esponenzialmente (emivita cfg.pca_hl ore); altrimenti
    finestra rettangolare. Usata sia dal backtest (rolling_pca_weights) sia dal bot live.
    """
    k = win.shape[1]
    ia, ib = cfg.assets.index(cfg.leg_a), cfg.assets.index(cfg.leg_b)
    if getattr(cfg, "ewma", False):
        a = 0.5 ** (np.arange(len(win))[::-1] / cfg.pca_hl)         # peso 1 sull'ultima barra, decade all'indietro
        a = a / a.sum()
        mu = a @ win
        X = win - mu
        cov = (X * a[:, None]).T @ X
        s = np.sqrt(np.diag(cov))
        C = cov / np.outer(s, s)
    else:
        mu, s = win.mean(0), win.std(0, ddof=1)
        C = np.corrcoef((win - mu) / s, rowvar=False)
    v = np.linalg.eigh(C)[1][:, -1]
    w = np.zeros(k)
    w[ia], w[ib] = 1.0, -1.0
    w -= (v[ia] - v[ib]) * v
    wr = w / s                          # pesi in spazio rendimenti (dollari)
    return wr / np.abs(wr).sum()


def rolling_pca_weights(rets: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Riga t = pesi calcolati sulla finestra [t-W, t-1], usabili per la barra t (causale)."""
    R = rets[list(cfg.assets)].values
    n, k = R.shape
    W = np.full((n, k), np.nan)
    for t in range(cfg.pca_win, n):
        W[t] = pca_weights_window(R[t - cfg.pca_win:t], cfg)
    return pd.DataFrame(W, index=rets.index, columns=cfg.assets)


def causal_garch_vol(y: pd.Series, cfg: Cfg) -> pd.Series:
    """sigma_t one-step-ahead: parametri stimati solo sul passato, refit settimanale."""
    yb = y * 1e4
    vol = pd.Series(np.nan, index=y.index)
    first = yb.first_valid_index()
    p0 = yb.index.get_loc(first)
    for start in range(p0 + cfg.garch_min_train, len(y), cfg.garch_refit):
        end = min(start + cfg.garch_refit, len(y))
        tr0 = max(p0, start - cfg.garch_train_win)
        res = arch_model(yb.iloc[tr0:start], mean="Zero", vol="GARCH", p=1, q=1,
                         dist="t").fit(disp="off")
        fixed = arch_model(yb.iloc[tr0:end], mean="Zero", vol="GARCH", p=1, q=1,
                           dist="t").fix(res.params)
        cv = fixed.conditional_volatility          # sigma_k dipende solo da y_{<k}
        vol.iloc[start:end] = cv.iloc[start - tr0:].values / 1e4
    return vol


def build_signal_and_features(rets: pd.DataFrame, cfg: Cfg):
    W = rolling_pca_weights(rets, cfg)
    pnl = (W * rets[list(cfg.assets)]).sum(axis=1, min_count=len(cfg.assets))
    L = pnl.cumsum()
    if getattr(cfg, "ewma", False):                                  # z-score con media/dev.std esponenziali
        ew = L.ewm(halflife=cfg.z_hl, min_periods=cfg.z_win)
        z = (L - ew.mean()) / ew.std()
    else:
        z = (L - L.rolling(cfg.z_win).mean()) / L.rolling(cfg.z_win).std()
    vol = causal_garch_vol(pnl, cfg)

    dL, Llag = L.diff(), L.shift(1)
    ou_b = dL.rolling(168).cov(Llag) / Llag.rolling(168).var()      # <0 => mean reverting
    var1 = pnl.rolling(168).var()
    vr12 = pnl.rolling(12).sum().rolling(168).var() / (12 * var1)    # <1 => mean reverting

    F = pd.DataFrame({
        "z": z,
        "vol_ratio": vol / vol.rolling(168).median(),
        "btc_trend_24h": rets[cfg.leg_a].rolling(24).sum(),
        "mom12": L.diff(12) / (vol * np.sqrt(12)),
        "ou_b": ou_b,
        "vr12": vr12,
    })
    return W, pnl, vol, F


# ------------------------------------------------------------------ triple barrier
def barrier_outcomes(event_pos, rets, W, vol, z, cfg: Cfg) -> pd.DataFrame:
    A = rets[list(cfg.assets)].values
    S = np.cumsum(A, axis=0)
    Wv, volv, zv = W.values, vol.values, z.values
    n, H = len(A), cfg.horizon
    rows = []
    for t in event_pos:
        e = t + cfg.entry_delay                      # entrata al close della barra e
        if e + H + 1 >= n:
            continue
        w, sig = Wv[e + 1], volv[e + 1]              # noti con dati fino a e
        if np.isnan(w).any() or np.isnan(sig):
            continue
        d = -np.sign(zv[t])
        path = d * ((S[e + 1:e + H + 1] - S[e]) @ w)     # PnL lordo ad ogni close
        tp = cfg.tp_mult * sig * np.sqrt(H)
        sl = cfg.sl_mult * sig * np.sqrt(H)
        i_tp = np.flatnonzero(path >= tp)
        i_sl = np.flatnonzero(path <= -sl)
        i_tp = i_tp[0] if len(i_tp) else H
        i_sl = i_sl[0] if len(i_sl) else H
        if i_tp == H and i_sl == H:
            kind, k = "time", H - 1
        elif i_tp < i_sl:
            kind, k = "tp", i_tp
        else:
            kind, k = "sl", i_sl
        gross = path[k]                               # valore reale, incluso overshoot
        exit_bps = cfg.exit_tp_bps if kind == "tp" else cfg.exit_sl_bps
        cost = (cfg.entry_bps + exit_bps + 2 * cfg.slip_bps + cfg.funding_bps_h * (k + 1)) / 1e4
        rows.append((t, e + 1 + k, kind, gross, gross - cost, k + 1))
    out = pd.DataFrame(rows, columns=["t_pos", "exit_pos", "kind", "gross", "net", "hold"])
    out["t0"] = rets.index[out["t_pos"].values]
    out["t1"] = rets.index[out["exit_pos"].values]
    out["y"] = (out["net"] > 0).astype(int)
    return out.set_index("t0")


def uniqueness_weights(out: pd.DataFrame, n: int) -> np.ndarray:
    conc = np.zeros(n + 1)
    for a, b in zip(out["t_pos"], out["exit_pos"]):
        conc[a:b + 1] += 1
    return np.array([(1.0 / conc[a:b + 1]).mean() for a, b in zip(out["t_pos"], out["exit_pos"])])


# ---------------------------------------------------------- ML con purged K-fold
def make_model(seed):
    return xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                             random_state=seed)


def purged_oof(X, y, t0, t1, w, cfg: Cfg):
    oof = np.full(len(X), np.nan)
    emb = pd.Timedelta(hours=cfg.horizon)
    for te in np.array_split(np.arange(len(X)), cfg.n_splits):
        ts, tend = t0.iloc[te].min(), t1.iloc[te].max() + emb
        tr = ((t1 < ts) | (t0 > tend)).values
        if tr.sum() < 30 or y[tr].nunique() < 2:
            continue
        m = make_model(cfg.seed).fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


# -------------------------------------------------------- simulazione sequenziale
def simulate_sequential(out: pd.DataFrame, probs: np.ndarray, th: float, cfg: Cfg) -> pd.DataFrame:
    """Una posizione per volta; cooldown dopo SL. Eventi gia' ordinati per t0."""
    taken, free_at = [], pd.Timestamp.min.tz_localize(out.index.tz)
    for i, (t0, r) in enumerate(out.iterrows()):
        if np.isnan(probs[i]) or probs[i] < th or t0 < free_at:
            continue
        taken.append(i)
        free_at = r["t1"] + (pd.Timedelta(hours=cfg.cooldown_after_sl) if r["kind"] == "sl"
                             else pd.Timedelta(0))
    return out.iloc[taken]


def perf(trades: pd.DataFrame, start, end, n_boot=2000, seed=0) -> dict:
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    daily = trades.groupby(trades["t1"].dt.floor("D"))["net"].sum().reindex(days, fill_value=0.0)
    sh = daily.mean() / (daily.std() + 1e-12) * np.sqrt(365)
    eq = daily.cumsum()
    rng = np.random.default_rng(seed)
    d, blk, bs = daily.values, 7, []
    for _ in range(n_boot):
        idx = np.concatenate([(rng.integers(0, len(d)) + np.arange(blk)) % len(d)
                              for _ in range(int(np.ceil(len(d) / blk)))])[:len(d)]
        s = d[idx]
        bs.append(s.mean() / (s.std() + 1e-12) * np.sqrt(365))
    return {"n_trades": len(trades),
            "win_rate": float((trades["net"] > 0).mean()) if len(trades) else np.nan,
            "mean_net_bps": float(trades["net"].mean() * 1e4) if len(trades) else np.nan,
            "mean_gross_bps": float(trades["gross"].mean() * 1e4) if len(trades) else np.nan,
            "total_net": float(trades["net"].sum()),
            "max_dd": float((eq.cummax() - eq).max()),
            "sharpe": float(sh), "sharpe_ci90": tuple(np.percentile(bs, [5, 95]).round(2))}


# --------------------------------------------------------------------- pipeline
def run_pipeline(prices: pd.DataFrame, cfg: Cfg = Cfg(), verbose=True) -> dict:
    rets = np.log(prices[list(cfg.assets)]).diff().dropna()
    W, pnl, vol, F = build_signal_and_features(rets, cfg)

    ok = F.notna().all(axis=1).values & (F["z"].abs() >= cfg.z_entry).values
    out = barrier_outcomes(np.flatnonzero(ok), rets, W, vol, F["z"], cfg)
    X = F.loc[out.index]
    y = out["y"]
    null_tp = cfg.sl_mult / (cfg.tp_mult + cfg.sl_mult)

    # split temporale: train = eventi con uscita prima dello split (purge), test dopo embargo
    t_split = out.index[int(len(out) * (1 - cfg.test_frac))]
    tr = (out["t1"] < t_split).values
    te = (out.index >= t_split + pd.Timedelta(hours=cfg.horizon))
    Xtr, ytr, otr = X[tr], y[tr], out[tr]
    Xte, yte, ote = X[te], y[te], out[te]
    wtr = uniqueness_weights(otr, len(rets))

    # OOF sul train -> soglia (simulazione sequenziale con costi)
    oof = purged_oof(Xtr, ytr, otr.index.to_series(), otr["t1"], wtr, cfg)
    best_th, best_pnl = 1.1, -np.inf
    for th in np.linspace(0.30, 0.70, 41):
        sim = simulate_sequential(otr, oof, th, cfg)
        if len(sim) >= 20 and sim["net"].sum() > best_pnl:
            best_th, best_pnl = th, sim["net"].sum()

    model = make_model(cfg.seed).fit(Xtr, ytr, sample_weight=wtr)
    p_te = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te) if yte.nunique() > 1 else np.nan

    start, end = ote.index.min(), ote["t1"].max()
    base = simulate_sequential(ote, np.ones(len(ote)), 0.0, cfg)           # solo z-score
    filt = simulate_sequential(ote, p_te, best_th, cfg)                    # meta-labeling
    res = {"null_tp_prob": null_tp, "hit_rate_tp_test": float((ote["kind"] == "tp").mean()),
           "threshold": best_th, "auc_test": auc, "n_events": len(out),
           "base": perf(base, start, end), "meta": perf(filt, start, end),
           "diag_ou_b_median": float(F["ou_b"].median()), "diag_vr12_median": float(F["vr12"].median())}
    if verbose:
        print(f"Eventi: {len(out)} | train {len(otr)} | test {len(ote)} | P(TP) sotto martingale: {null_tp:.2f}")
        print(f"Soglia (OOF train): {best_th:.2f} | AUC test: {auc:.3f} | "
              f"mediana ou_b: {res['diag_ou_b_median']:.4f} | mediana VR12: {res['diag_vr12_median']:.2f}")
        for k in ("base", "meta"):
            print(f"  {k:5s}", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k].items()})
    return res


# ------------------------------------------------------------ dati sintetici (test)
def simulate_prices(n=7000, mean_revert=False, seed=0, half_life=12, ou_innov=0.003):
    rng = np.random.default_rng(seed)
    t_ = lambda size: rng.standard_t(5, size) / np.sqrt(5 / 3)
    f, h = np.zeros(n), np.full(n, 0.004 ** 2)
    for t in range(1, n):
        h[t] = 2e-7 + 0.07 * f[t - 1] ** 2 + 0.90 * h[t - 1]
        f[t] = np.sqrt(h[t]) * t_(1)[0]
    beta, idio = np.array([1.0, 1.25, 1.4]), np.array([0.0008, 0.0010, 0.0020])
    r = f[:, None] * beta + t_((n, 3)) * idio
    if mean_revert:                                  # componente OU stazionaria tra ETH e BTC
        phi, u = 0.5 ** (1 / half_life), np.zeros(n)
        for t in range(1, n):
            u[t] = phi * u[t - 1] + ou_innov * t_(1)[0]
        r[1:, 1] += np.diff(u)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=["BTC", "ETH", "SOL"])


if __name__ == "__main__":
    cfg = Cfg()
    print("=== PLACEBO: random walk, nessuna mean reversion (atteso: AUC~0.5, Sharpe <= 0) ===")
    run_pipeline(simulate_prices(mean_revert=False, seed=1), cfg)
    print("\n=== CONTROLLO POSITIVO: OU iniettato tra ETH e BTC (atteso: edge rilevabile) ===")
    run_pipeline(simulate_prices(mean_revert=True, seed=1), cfg)
