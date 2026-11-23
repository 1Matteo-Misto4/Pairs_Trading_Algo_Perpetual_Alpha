"""
hl_connect.py -- livello di connessione alla venue (Hyperliquid TESTNET, dex `par`).

Regole dalla guida degli organizzatori che questo modulo applica:
  * ogni query "market-wide" e di posizione passa `dex="par"`; i coin hanno prefisso `par:`
  * prezzi: max 5 cifre significative E max (6 - szDecimals) decimali; size: multiplo del lot
  * minimo $10 per ordine; `status: ok` a livello di richiesta puo' nascondere ordini rifiutati
    -> si leggono SEMPRE i singoli `statuses`
  * le query usano l'indirizzo del TEAM WALLET; le azioni firmate usano l'API wallet

Le letture usano `info.post("/info", payload)` con i `type` esatti della guida, cosi' non dipendono
dai nomi dei parametri dei metodi dell'SDK. Le azioni firmate usano l'SDK ufficiale 0.24.0.
"""
import json
import logging
import math
import time

import numpy as np
import pandas as pd

BASE_URL = "https://api.hyperliquid-testnet.xyz"
DEX = "par"
log = logging.getLogger("venue")


def coin(a: str) -> str:
    return a if ":" in a else f"{DEX}:{a}"


def fmt_price(px: float, sz_decimals: int) -> float:
    """Max 5 cifre significative e max (6 - szDecimals) decimali."""
    px = float(f"{px:.5g}")
    return round(px, max(0, 6 - sz_decimals))


def fmt_size(sz: float, sz_decimals: int) -> float:
    """Arrotonda PER DIFETTO al lot 10^-szDecimals."""
    f = 10 ** sz_decimals
    return math.floor(sz * f + 1e-9) / f


def parse_statuses(res, n):
    """Trasforma la risposta di un ordine in una lista di esiti per singola operazione."""
    if not isinstance(res, dict) or res.get("status") != "ok":
        return [{"filled": 0.0, "avg_px": None, "error": f"request failed: {res}"}] * n
    out = []
    for st in res["response"]["data"]["statuses"]:
        if "filled" in st:
            out.append({"filled": float(st["filled"]["totalSz"]),
                        "avg_px": float(st["filled"]["avgPx"]), "error": None})
        elif "resting" in st:
            out.append({"filled": 0.0, "avg_px": None, "error": "resting (inatteso per IOC)",
                        "oid": st["resting"]["oid"]})
        else:
            out.append({"filled": 0.0, "avg_px": None, "error": st.get("error", str(st))})
    return out


def walk_cost_bps(levels, notional, oracle, side):
    """Slippage in bps contro l'ORACOLO per eseguire `notional` $ a mercato lungo `levels`.
    NaN se il book non basta. side: 'buy' (si consumano gli ask) o 'sell' (si consumano i bid)."""
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
        return float("nan")
    avg = spent / qty
    return (avg - oracle) / oracle * 1e4 if side == "buy" else (oracle - avg) / oracle * 1e4


class Venue:
    def __init__(self, team_address: str, api_key: str | None = None, dry_run: bool = True):
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        self.addr, self.dry_run = team_address, dry_run
        self.info = Info(BASE_URL, skip_ws=True, perp_dexs=[DEX])
        self.exchange = None
        if api_key:
            wallet = Account.from_key(api_key)
            self.exchange = Exchange(wallet, BASE_URL, account_address=team_address, perp_dexs=[DEX])
        self.load_meta()

    # ------------------------------------------------------------------ letture
    def _post(self, payload):
        return self.info.post("/info", payload)

    def load_meta(self):
        m = self._post({"type": "meta", "dex": DEX})
        self.sz_dec = {coin(u["name"]): int(u["szDecimals"]) for u in m["universe"]}
        self.max_lev = {coin(u["name"]): int(u["maxLeverage"]) for u in m["universe"]}

    def ctxs(self) -> pd.DataFrame:
        """oraclePx, markPx, midPx, funding, openInterest per ogni mercato `par`."""
        meta, ctx = self._post({"type": "metaAndAssetCtxs", "dex": DEX})
        df = pd.DataFrame(ctx, index=[coin(u["name"]) for u in meta["universe"]])
        for c in ("oraclePx", "markPx", "midPx", "funding", "openInterest"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        return df

    def oracle_prices(self, assets) -> pd.Series:
        df = self.ctxs()
        return pd.Series({a: float(df.loc[coin(a), "oraclePx"]) for a in assets})

    def l2(self, c: str, band_bps: float = 10.0) -> dict:
        bids, asks = self._post({"type": "l2Book", "coin": c})["levels"]
        if not bids or not asks:
            return dict(bid=np.nan, ask=np.nan, bid_sz=np.nan, ask_sz=np.nan,
                        depth_bid_usd=0.0, depth_ask_usd=0.0)
        bid, ask = float(bids[0]["px"]), float(asks[0]["px"])
        mid = (bid + ask) / 2
        db = sum(float(l["px"]) * float(l["sz"]) for l in bids if float(l["px"]) >= mid * (1 - band_bps / 1e4))
        da = sum(float(l["px"]) * float(l["sz"]) for l in asks if float(l["px"]) <= mid * (1 + band_bps / 1e4))
        return dict(bid=bid, ask=ask, bid_sz=float(bids[0]["sz"]), ask_sz=float(asks[0]["sz"]),
                    depth_bid_usd=db, depth_ask_usd=da)

    def l2_levels(self, c: str):
        """(bids, asks) come liste di dict px/sz, dal migliore al peggiore."""
        bids, asks = self._post({"type": "l2Book", "coin": c})["levels"]
        return bids, asks

    def positions(self) -> dict:
        st = self._post({"type": "clearinghouseState", "user": self.addr, "dex": DEX})
        return {coin(ap["position"]["coin"]): float(ap["position"]["szi"])
                for ap in st.get("assetPositions", []) if float(ap["position"]["szi"]) != 0}

    def account_value(self) -> float:
        """USDPR spot (margine unificato) + PnL non realizzato dei perp `par`. Da verificare in Practice."""
        sp = self._post({"type": "spotClearinghouseState", "user": self.addr})
        bal = sum(float(b["total"]) for b in sp.get("balances", []) if b["coin"] == "USDPR")
        st = self._post({"type": "clearinghouseState", "user": self.addr, "dex": DEX})
        upnl = sum(float(ap["position"].get("unrealizedPnl", 0)) for ap in st.get("assetPositions", []))
        return bal + upnl

    def perp_limits(self):
        """Payload grezzo di perpDexLimits: il formato va ispezionato al primo avvio."""
        return self._post({"type": "perpDexLimits", "dex": DEX})

    def user_rate_limit(self):
        return self._post({"type": "userRateLimit", "user": self.addr})

    # ------------------------------------------------------------------ azioni firmate
    def set_leverage(self, c: str, lev: int):
        lev = min(lev, self.max_lev[c])
        if self.dry_run or self.exchange is None:
            log.info("DRY set_leverage %s %sx", c, lev)
            return None
        return self.exchange.update_leverage(lev, c, True)       # cross margin (account unificato)

    def ioc_batch(self, legs, slip_bps: float, reduce_only: bool = False):
        """
        legs: lista di dict(coin, is_buy, sz, ref_px). Un ordine IOC per gamba, inviati in UN
        solo bulk_orders. Il limite di prezzo e' ref_px +/- slip_bps (NON lo slippage 5% di default).
        Restituisce un esito per gamba: filled, avg_px, error.
        """
        reqs = []
        for l in legs:
            sd = self.sz_dec[l["coin"]]
            px = l["ref_px"] * (1 + slip_bps / 1e4 if l["is_buy"] else 1 - slip_bps / 1e4)
            reqs.append({"coin": l["coin"], "is_buy": bool(l["is_buy"]),
                         "sz": fmt_size(l["sz"], sd), "limit_px": fmt_price(px, sd),
                         "order_type": {"limit": {"tif": "Ioc"}}, "reduce_only": reduce_only})
        if self.dry_run or self.exchange is None:
            log.info("DRY ordini: %s", json.dumps(reqs))
            return [{"filled": r["sz"], "avg_px": l["ref_px"], "error": None, "simulated": True}
                    for r, l in zip(reqs, legs)]
        res = self.exchange.bulk_orders(reqs)
        log.info("ordini: %s -> %s", json.dumps(reqs), json.dumps(res)[:600])
        return parse_statuses(res, len(reqs))
