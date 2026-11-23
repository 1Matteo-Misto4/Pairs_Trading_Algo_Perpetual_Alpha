"""
setup_account.py -- da eseguire UNA SOLA VOLTA, a mano, da UNA persona del team.

Cosa fa (guida "Set up the account"):
  1. controlla che il wallet del team sia finanziato (USDPR)
  2. imposta la modalita' unificata (user_set_abstraction -> "unifiedAccount")
  3. approva API wallet separati (approve_agent): "trader" per il bot, "watchdog" per il processo di
     sicurezza. Un API wallet PER PROCESSO evita collisioni di nonce. Possono solo fare trading:
     non possono prelevare.
  4. salva le chiavi degli API wallet in .secrets/<nome>.key (permessi 600), MAI nel codice o nei log

SICUREZZA
  * la chiave PRIVATA del team wallet viene chiesta da tastiera (getpass) e non viene salvata ne' stampata
  * non incollare mai chiavi in chat, Discord o Git. `.secrets/` e' nel .gitignore
  * il bot usa solo la chiave dell'API wallet, mai quella del team wallet

Uso:   export TEAM_ADDRESS=0x...   (indirizzo PUBBLICO)   &&   python setup_account.py
"""
import getpass
import inspect
import os
import stat
import time
from pathlib import Path

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from dotenv import load_dotenv
load_dotenv()

BASE_URL = "https://api.hyperliquid-testnet.xyz"
DEX = "par"
AGENT_NAMES = ("trader", "watchdog")
EXPIRY_DAYS = 30


def main():
    team = os.environ["TEAM_ADDRESS"]
    info = Info(BASE_URL, skip_ws=True, perp_dexs=[DEX])

    key = getpass.getpass("Chiave privata del TEAM WALLET (non viene salvata): ").strip()
    wallet = Account.from_key(key)
    assert wallet.address.lower() == team.lower(), "La chiave non corrisponde a TEAM_ADDRESS"
    ex = Exchange(wallet, BASE_URL, account_address=team, perp_dexs=[DEX])

    sp = info.spot_user_state(team)
    print("Saldi:", [(b["coin"], b["total"]) for b in sp.get("balances", [])])

    print("Modalita' account attuale:", info.query_user_abstraction_state(team))
    print("Firma user_set_abstraction:", inspect.signature(ex.user_set_abstraction))
    print("Imposto unifiedAccount ->", ex.user_set_abstraction(team, "unifiedAccount"))
    print("Modalita' account ora:", info.query_user_abstraction_state(team))

    out = Path(".secrets")
    out.mkdir(exist_ok=True)
    valid_until = int((time.time() + EXPIRY_DAYS * 86400) * 1000)
    for name in AGENT_NAMES:
        # convenzione Hyperliquid per la scadenza: "<nome> valid_until <ms>" -- verifica con extraAgents
        res, agent_key = ex.approve_agent(f"{name} valid_until {valid_until}")
        print(f"approve_agent[{name}] ->", res)
        path = out / f"{name}.key"
        path.write_text(agent_key)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)         # 600
        print(f"  chiave salvata in {path} (non stamparla, non committarla)")

    print("API wallet approvati:", info.post("/info", {"type": "extraAgents", "user": team}))
    print("Fatto. Il bot userà .secrets/trader.key con account_address = TEAM_ADDRESS.")


if __name__ == "__main__":
    main()
