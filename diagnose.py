"""
diagnose.py -- diagnosi dell'edge a diversi z_entry. NON invia ordini, non usa chiavi, non tocca bundle.

Per ogni z_entry stampa:
  1. eventi, win rate e rendimento netto MEDIO della strategia di base (solo z-score, nessun modello)
     e la composizione delle uscite TP/SL/time-out. Sotto martingala P(TP) atteso = sl/(tp+sl) = 0.40.
  2. AUC OOF pooled e AUC / win rate PER PEZZO (se l'AUC pooled e' basso solo per colpa del cambio di
     regime tra pezzi, i pezzi singoli stanno sopra 0.5 e il win rate cambia molto da pezzo a pezzo).
  3. evaluate() di v3: train/test temporale, test usato una volta: base (solo z) vs meta (con modello).

Uso:   python diagnose.py --data-dir data/organizers [--z 1.0 1.2 1.5]
"""
import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_loader import load_history
from pairs_pipeline_v2 import purged_oof, uniqueness_weights
from pairs_pipeline_v3 import PEERS, Cfg, CostModel, barrier_events, entry_mask, evaluate, prepare


def diagnose(prices, zs, cfg0: Cfg, cm: CostModel):
    prep = prepare(prices, cfg0)                      # parte pesante (PCA + GARCH): indipendente da z_entry
    rets, W, vol, F, Pm = prep["rets"], prep["W"], prep["vol"], prep["F"], prep["Pm"]
    null_tp = cfg0.sl_mult / (cfg0.tp_mult + cfg0.sl_mult)
    for z in zs:
        cfg = replace(cfg0, z_entry=z)
        print(f"\n================ z_entry = {z} ================")
        ok = entry_mask(F, cfg)
        out = barrier_events(np.flatnonzero(ok), Pm, rets, W, vol, F["z"], cfg, cm, vr=F["vol_ratio"].values)
        X, y = F.loc[out.index], out["y"]
        share = out["kind"].value_counts(normalize=True).round(3).to_dict()
        print(f"[1] eventi {len(out)} | win rate netto {y.mean():.3f} | rend. netto medio {out['net'].mean()*1e4:6.1f} bps "
              f"(lordo {out['gross'].mean()*1e4:6.1f} bps)")
        print(f"    uscite {share} | P(TP) osservata {share.get('tp', 0):.3f} vs {null_tp:.2f} sotto martingala")

        w = uniqueness_weights(out, len(rets))
        oof = purged_oof(X, y, out.index.to_series(), out["t1"], w, cfg)
        have = ~np.isnan(oof)
        auc = roc_auc_score(y[have], oof[have]) if y[have].nunique() > 1 else np.nan
        print(f"[2] AUC OOF pooled {auc:.3f}  |  per pezzo (ordine cronologico):")
        for k, te in enumerate(np.array_split(np.arange(len(X)), cfg.n_splits)):
            m = ~np.isnan(oof[te])
            yt, pt = y.iloc[te][m], oof[te][m]
            a = roc_auc_score(yt, pt) if m.sum() > 10 and yt.nunique() > 1 else np.nan
            print(f"      pezzo {k}: {out.index[te[0]].date()} -> {out.index[te[-1]].date()} | n={m.sum():4d} | "
                  f"win rate {yt.mean():.3f} | AUC {a:.3f}")

        print("[3] holdout temporale (test 30% finale, usato una volta):")
        evaluate(prep, cfg, cm, PEERS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--z", type=float, nargs="+", default=[1.0, 1.2, 1.5])
    a = ap.parse_args()
    cfg, cm = Cfg(), CostModel()
    diagnose(load_history(a.data_dir, cfg.assets), a.z, cfg, cm)
