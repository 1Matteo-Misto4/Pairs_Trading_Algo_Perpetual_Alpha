"""
train_model.py -- addestra il modello finale (meta-labeling) sullo storico e lo salva per il bot.

ATTENZIONE: qui il modello e la soglia usano TUTTO lo storico (soglia scelta su probabilita' OOF con
purged CV). Serve a far girare il bot in Practice. La validazione onesta resta `evaluate()` di v3
(train/test temporale con test usato una volta sola): non usare i numeri di questo script come prova di edge.

Uso:
    python train_model.py --data-dir data/organizers --out models/bundle.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from data_loader import load_history
from pairs_pipeline_v2 import make_model, purged_oof, uniqueness_weights
from pairs_pipeline_v3 import Cfg, CostModel, barrier_events, entry_mask, prepare, run_sim


def fit_final(prices, cfg: Cfg, cm: CostModel, out_path=None, verbose=True):
    prep = prepare(prices, cfg)
    rets, W, vol, F, Pm = prep["rets"], prep["W"], prep["vol"], prep["F"], prep["Pm"]
    ok = entry_mask(F, cfg)
    out = barrier_events(np.flatnonzero(ok), Pm, rets, W, vol, F["z"], cfg, cm, vr=F["vol_ratio"].values)
    X, y = F.loc[out.index], out["y"]
    w = uniqueness_weights(out, len(rets))

    oof = purged_oof(X, y, out.index.to_series(), out["t1"], w, cfg)
    idx = rets.index
    rates = np.zeros((len(idx), len(cfg.assets)))
    s0, e0 = int(out["t_pos"].min()), int(out["exit_pos"].max())
    best_th, best_sh = 1.1, -np.inf
    for th in np.linspace(0.30, 0.70, 41):
        m = run_sim(out, oof, th, cfg, cm, idx, rates, s0, e0)
        if m["n_trades"] >= 20 and m["sharpe"] > best_sh:
            best_th, best_sh = th, m["sharpe"]

    have = ~np.isnan(oof)
    auc = roc_auc_score(y[have], oof[have]) if y[have].nunique() > 1 else np.nan
    model = make_model(cfg.seed).fit(X, y, sample_weight=w)
    bundle = dict(model=model, threshold=float(best_th), features=list(F.columns), cfg=cfg,
                  oof_auc=float(auc), n_events=len(out), trained_to=str(idx[-1]))
    if verbose:
        print(f"Eventi {len(out)} | soglia {best_th:.2f} | AUC OOF {auc:.3f} | storico fino a {idx[-1]}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)
        print("Salvato", out_path)
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="models/bundle.joblib")
    ap.add_argument("--z-entry", type=float, default=1.0,
                    help="soglia |z| di ingresso con cui costruire gli eventi e addestrare (default 1.0; originale 1.2)")
    ap.add_argument("--ewma", action="store_true", help="z-score e PCA esponenziali")
    ap.add_argument("--dyn-z", action="store_true", help="soglia z dinamica sulla volatilita'")
    ap.add_argument("--dyn-size", action="store_true", help="size dinamica (influenza la soglia scelta sul simulatore)")
    ap.add_argument("--skip-vol-ratio", type=float, default=0.0)
    a = ap.parse_args()
    cfg, cm = Cfg(z_entry=a.z_entry, ewma=a.ewma, dyn_z=a.dyn_z, dyn_size=a.dyn_size, skip_vol_ratio=a.skip_vol_ratio), CostModel()
    print(f"z_entry = {cfg.z_entry}")
    fit_final(load_history(a.data_dir, cfg.assets), cfg, cm, a.out)
