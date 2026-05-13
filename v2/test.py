import os
from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv

# Carica il file .env
load_dotenv()

key = os.getenv('ALPACA_API_KEY')
secret = os.getenv('ALPACA_SECRET_KEY')

print(f"Key trovata: {key[:4]}... (lunghezza: {len(key) if key else 0})")

try:
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    # Facciamo una piccola richiesta per testare l'autenticazione
    from alpaca.data.requests import StockLatestQuoteRequest
    req = StockLatestQuoteRequest(symbol_or_symbols="AAPL")
    quote = client.get_stock_latest_quote(req)
    print("✅ Autenticazione riuscita! Quote AAPL:", quote["AAPL"].ask_price)
except Exception as e:
    print("❌ Errore:", e)
