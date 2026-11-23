from hyperliquid.info import Info
from hyperliquid.utils import constants

# Indirizzo del tuo team
wallet_address = "0xaE8d363b4c35f2Aea80F5C49D9ae136B17b51aA2"

print(f"Interrogazione del portafoglio per: {wallet_address}...\n")
info = Info(constants.MAINNET_API_URL, skip_ws=True)
user_state = info.user_state(wallet_address)

# Estrae il valore totale del portafoglio (Margine + PnL)
margin_summary = user_state.get("marginSummary", {})
total_value = margin_summary.get("accountValue", "N/D")
print(f"💰 VALORE TOTALE PORTAFOGLIO: {total_value} $\n")

# Estrae le posizioni aperte
positions = user_state.get("assetPositions", [])
if not positions:
    print("Nessuna posizione aperta al momento.")
else:
    print("📊 POSIZIONI APERTE:")
    for pos in positions:
        coin = pos['position']['coin']
        size = pos['position']['szi']
        entry_price = pos['position']['entryPx']
        pnl = pos['position']['unrealizedPnl']
        print(f" - {coin}: Quantità {size} | Prezzo Ingresso {entry_price}$ | PnL (Profitto/Perdita): {pnl}$")