"""
data_loader.py -- carica le candele orarie degli organizzatori (un CSV per mercato) e le rende
utilizzabili dalla pipeline: DataFrame di prezzi di chiusura, indice UTC orario, colonne = asset.

Lo schema dei CSV non e' noto a priori: il loader riconosce le colonne piu' comuni e stampa cio' che
ha capito. VERIFICA a occhio i primi/ultimi timestamp: assumiamo che il timestamp sia l'ORA DI APERTURA
della candela e che `close` sia il prezzo a fine ora (convenzione Hyperliquid, campi t e c).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLS = ("timestamp", "time", "t", "datetime", "date", "open_time", "opentime", "open time")
CLOSE_COLS = ("close", "c", "closing_price", "close_price")


def _parse_time(s: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.iloc[0] > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True)


def load_market_csv(path) -> pd.Series:
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    tcol = next((low[c] for c in TIME_COLS if c in low), None)
    ccol = next((low[c] for c in CLOSE_COLS if c in low), None)
    if tcol is None or ccol is None:
        raise ValueError(f"{path}: colonne non riconosciute {list(df.columns)}")
    s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").values, index=_parse_time(df[tcol]))
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s.index = s.index.floor("h")
    return s


def load_history(folder, assets=("BTC", "ETH", "HYPE"), max_ffill=3) -> pd.DataFrame:
    folder = Path(folder)
    cols = {}
    for a in assets:
        files = [f for f in folder.glob("*.csv") if re.search(rf"(?<![A-Za-z]){a}(?![A-Za-z])", f.stem, re.I)]
        if len(files) != 1:
            raise FileNotFoundError(f"Atteso 1 CSV per {a} in {folder}, trovati: {[f.name for f in files]}")
        cols[a] = load_market_csv(files[0])
        print(f"{a}: {files[0].name}  {cols[a].index[0]} -> {cols[a].index[-1]}  ({len(cols[a])} barre)")
    px = pd.DataFrame(cols)
    px = px.asfreq("h")
    missing = int(px.isna().any(axis=1).sum())
    px = px.ffill(limit=max_ffill).dropna()
    gaps = px.index.to_series().diff().dropna()
    n_gaps = int((gaps > pd.Timedelta(hours=1)).sum())
    print(f"Barre totali {len(px)} | ore mancanti riempite/scartate {missing} | buchi residui {n_gaps}")
    return px


def backfill_mainnet(assets, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Candele orarie dalla API PUBBLICA di mainnet (stessa fonte dichiarata dagli organizzatori) per
    colmare il buco tra la fine dei CSV e adesso. Max 5000 candele per richiesta.
    NON TESTATO da qui (rete non raggiungibile): provalo in locale.
    """
    import requests
    cols = {}
    for a in assets:
        r = requests.post("https://api.hyperliquid.xyz/info", timeout=20, json={
            "type": "candleSnapshot",
            "req": {"coin": a, "interval": "1h", "startTime": int(start.timestamp() * 1000),
                    "endTime": int(end.timestamp() * 1000)}})
        r.raise_for_status()
        rows = r.json()
        cols[a] = pd.Series([float(x["c"]) for x in rows],
                            index=pd.to_datetime([x["t"] for x in rows], unit="ms", utc=True))
    return pd.DataFrame(cols).sort_index()
