"""
liquidity_report.py -- riassume la liquidita' misurata dal collector (data/collector/par_*.csv).
Solo lettura: non usa rete ne' chiavi.

Per ogni mercato stampa mediana, 10-esimo percentile (caso brutto) e minimo di:
  * profondita' entro 10 bps (bid e ask, in $)
  * distanza best bid/ask dall'oracolo (bps): e' il "mezzo spread vs oracolo" di CostModel
e li confronta con i valori PLACEHOLDER di CostModel (depth_usd, half_spread_bps).
Con --gross stima anche quante volte il book NON avrebbe coperto il notional per gamba.

Uso:   python liquidity_report.py [--folder data/collector] [--gross 2000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_pipeline_v3 import CostModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/collector")
    ap.add_argument("--gross", type=float, default=2000.0, help="notional lordo del bot (default 2000)")
    a = ap.parse_args()

    files = sorted(list(Path(a.folder).glob("par_*.csv")) + list(Path(a.folder).glob("mkt_*.csv")))
    if not files:
        raise SystemExit(f"Nessun file par_*.csv o mkt_*.csv in {a.folder}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["depth_min"] = df[["depth_bid_usd", "depth_ask_usd"]].min(axis=1)
    df["half_spread_oracle"] = df[["bid_vs_oracle_bps", "ask_vs_oracle_bps"]].max(axis=1)
    print(f"{len(files)} file | {len(df)} righe | da {df['ts'].min()} a {df['ts'].max()}")

    cm = CostModel()
    ph = {f"par:{a_}": (d, s) for a_, d, s in zip(("BTC", "ETH", "HYPE"), cm.depth_usd, cm.half_spread_bps)}
    leg_notional = a.gross * 0.5            # BTC/ETH ~ meta' del gross ciascuna

    for c, g in df.groupby("coin"):
        d, s = g["depth_min"], g["half_spread_oracle"]
        pd_, ps_ = ph.get(c, (np.nan, np.nan))
        print(f"\n=== {c}  ({len(g)} snapshot) ===")
        print(f"  profondita' entro 10bps (lato piu' sottile): mediana ${d.median():,.0f} | p10 ${d.quantile(.1):,.0f} "
              f"| min ${d.min():,.0f}   [placeholder CostModel ${pd_:,.0f}]")
        print(f"  distanza dall'oracolo (lato peggiore):       mediana {s.median():5.1f} bps | p90 {s.quantile(.9):5.1f} "
              f"| max {s.max():5.1f}   [placeholder CostModel {ps_} bps]")
        thin = (d < leg_notional).mean() * 100
        print(f"  snapshot con profondita' < ${leg_notional:,.0f} (notional per gamba a gross ${a.gross:,.0f}): {thin:.1f}%")
        if g["bid"].isna().any() or g["ask"].isna().any():
            print(f"  ATTENZIONE: {int(g['bid'].isna().sum())} snapshot con book vuoto da un lato")

    df["hour"] = df["ts"].dt.hour
    print("\nProfondita' mediana per ora UTC (lato piu' sottile, $):")
    print(df.pivot_table(index="hour", columns="coin", values="depth_min", aggfunc="median").round(0).to_string())


if __name__ == "__main__":
    main()
