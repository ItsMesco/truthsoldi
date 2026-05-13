import os
import csv
import asyncio
import pandas as pd
import torch
import joblib
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

from alpaca.data.live import NewsDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from dotenv import load_dotenv

# ==========================================
# 1. SETUP CREDENZIALI E CLIENT ALPACA
# ==========================================
load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY', '').strip().replace('"', '').replace("'", "")
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '').strip().replace('"', '').replace("'", "")

trade_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_stream = NewsDataStream(API_KEY, SECRET_KEY)

# ==========================================
# 2. PARAMETRI BOT
# ==========================================
MAX_OPEN_POSITIONS = 100
POSITION_TIMEOUT_MINUTES = 30
BUDGET_FRACTION = 0.10
MIN_ORDER_NOTIONAL = 200.0
MIN_STOP_LOSS_PCT = 0.008      # 0.8%
MIN_TAKE_PROFIT_PCT = 0.012    # 1.2%
TP_SL_RATIO = 1.5
SOGLIA_COMPRA = 0.55
SOGLIA_VENDI = 0.65

opened_at_by_symbol = {}

# ==========================================
# 3. CARICAMENTO MODELLI
# ==========================================
print("🧠 Caricamento di FinBERT in corso...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

print("🌲 Caricamento di XGBoost in corso...")
xgb_model = joblib.load("bot_brain_xgb_robust.pkl")
print("✅ Modelli pronti. In attesa di notizie dal mercato...")

# ==========================================
# 4. SUPPORTO
# ==========================================
def get_market_data(symbol):
    try:
        req_trade = StockLatestTradeRequest(symbol_or_symbols=symbol)
        latest_trade = data_client.get_stock_latest_trade(req_trade)
        current_price = float(latest_trade[symbol].price)

        req_bar = StockLatestBarRequest(symbol_or_symbols=symbol)
        latest_bar = data_client.get_stock_latest_bar(req_bar)
        bar = latest_bar[symbol]
        volatility = ((bar.high - bar.low) / bar.close) * 100.0 if bar.close else 0.0

        return current_price, volatility
    except Exception as e:
        print(f"⚠️ Impossibile ottenere dati completi per {symbol}: {e}")
        return None, 0.0


def analyze_sentiment(text):
    inputs = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    probs = softmax(outputs.logits, dim=-1)[0]
    return probs[0].item(), probs[1].item(), probs[2].item()


def get_open_positions_count():
    try:
        return len(trade_client.get_all_positions())
    except Exception as e:
        print(f"⚠️ Errore nel conteggio posizioni aperte: {e}")
        return 999


def calculate_qty(current_price):
    try:
        account = trade_client.get_account()
        cash = float(account.cash)
        capitale_trade = max(MIN_ORDER_NOTIONAL, cash * BUDGET_FRACTION)
        qty = int(capitale_trade / current_price)
        return max(0, qty), cash, capitale_trade
    except Exception as e:
        print(f"⚠️ Errore nel calcolo quantità: {e}")
        return 0, 0.0, 0.0


def build_tp_sl_prices(side, current_price, vol):
    vol_effettiva = max((vol / 100.0), MIN_STOP_LOSS_PCT)
    stop_loss_pct = vol_effettiva
    take_profit_pct = max(vol_effettiva * TP_SL_RATIO, MIN_TAKE_PROFIT_PCT)

    decimals = 2 if current_price >= 1.00 else 4
    min_tick = 0.01 if current_price >= 1.00 else 0.0001

    if side == OrderSide.BUY:
        take_profit_price = round(current_price * (1 + take_profit_pct), decimals)
        stop_loss_price = round(current_price * (1 - stop_loss_pct), decimals)
        take_profit_price = max(take_profit_price, round(current_price + min_tick, decimals))
        stop_loss_price = min(stop_loss_price, round(current_price - min_tick, decimals))
    else:
        take_profit_price = round(current_price * (1 - take_profit_pct), decimals)
        stop_loss_price = round(current_price * (1 + stop_loss_pct), decimals)
        take_profit_price = min(take_profit_price, round(current_price - min_tick, decimals))
        stop_loss_price = max(stop_loss_price, round(current_price + min_tick, decimals))

    return take_profit_price, stop_loss_price, stop_loss_pct, take_profit_pct


# ==========================================
# 5. TRADING
# ==========================================
def execute_bracket_trade(symbol, side, confidence, current_price, vol):
    if get_open_positions_count() >= MAX_OPEN_POSITIONS:
        print(f"⚠️ Limite massimo posizioni aperte raggiunto ({MAX_OPEN_POSITIONS}). Nessun nuovo trade.")
        return

    qty, cash, capitale_trade = calculate_qty(current_price)
    if qty < 1:
        print(f"⚠️ Cash insufficiente per aprire una posizione su {symbol}. Cash: ${cash:.2f}")
        return

    take_profit_price, stop_loss_price, stop_loss_pct, take_profit_pct = build_tp_sl_prices(side, current_price, vol)

    print(f"💰 ESECUZIONE {side.name} SU {symbol} a ${current_price:.4f} | Sicurezza AI: {confidence:.1f}%")
    print(f" 💵 Cash disponibile: ${cash:.2f} | Allocazione trade: ${capitale_trade:.2f} | Quantità: {qty}")
    print(f" 🎯 Take Profit: ${take_profit_price} ({take_profit_pct*100:.2f}%) | 🛡️ Stop Loss: ${stop_loss_price} ({stop_loss_pct*100:.2f}%)")

    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price)
        )
        trade_client.submit_order(order_data=req)
        opened_at_by_symbol[symbol] = datetime.now(timezone.utc)
        print("✅ Ordine Bracket inviato con successo!")
    except Exception as e:
        print(f"❌ Errore nell'invio dell'ordine: {e}")


async def close_old_positions():
    while True:
        try:
            clock = trade_client.get_clock()
            if clock.is_open:
                positions = trade_client.get_all_positions()
                now = datetime.now(timezone.utc)

                for position in positions:
                    symbol = position.symbol
                    opened_at = opened_at_by_symbol.get(symbol)

                    if opened_at is None:
                        try:
                            # Ottima pensata l'utilizzo di position.age
                            age_seconds = int(float(position.age))
                            opened_at = now - timedelta(seconds=age_seconds)
                        except Exception:
                            continue

                    if now - opened_at > timedelta(minutes=POSITION_TIMEOUT_MINUTES):
                        print(f"⏳ TEMPO SCADUTO ({POSITION_TIMEOUT_MINUTES} min) per {symbol}. Chiusura forzata.")
                        try:
                            # ----- LA MODIFICA SALVAVITA È QUI -----
                            # Aggiungiamo cancel_orders=True per distruggere i vecchi Take Profit/Stop Loss
                            trade_client.close_position(
                                symbol_or_asset_id=symbol, 
                                cancel_orders=True
                            )
                            # ---------------------------------------
                            
                            opened_at_by_symbol.pop(symbol, None)
                            print(f"   🧹 Posizione {symbol} chiusa con successo e ordini liberati.")
                        except Exception as e:
                            print(f"⚠️ Errore nella chiusura forzata di {symbol}: {e}")

                if len(positions) >= MAX_OPEN_POSITIONS:
                    print(f"⚠️ Attenzione: hai {len(positions)} posizioni aperte in contemporanea.")
        except Exception as e:
            print(f"⚠️ Errore nel monitoraggio temporale delle posizioni: {e}")

        await asyncio.sleep(60)


# ==========================================
# 6. NEWS HANDLER
# ==========================================
async def on_news(news):
    try:
        clock = trade_client.get_clock()
        mercato_aperto = clock.is_open
    except Exception as e:
        print(f"⚠️ Errore nel controllo dell'orologio: {e}")
        return

    symbols = getattr(news, 'symbols', []) or []
    headline = getattr(news, 'headline', '') or ''

    print(f"\n📰 NUOVA NEWS: '{headline}' (Mercato Aperto: {mercato_aperto})")

    for symbol in symbols:
        if symbol == '*' or len(symbol) > 5:
            continue

        print(f"🔍 Analisi per {symbol}...")

        if get_open_positions_count() >= MAX_OPEN_POSITIONS:
            print(f"⚠️ Troppe posizioni aperte. Ignoro {symbol} finché non si libera spazio.")
            continue

        try:
            trade_client.get_open_position(symbol)
            print(f" 🛡️ Hai già una posizione aperta su {symbol}. Ignoro la notizia.")
            continue
        except Exception:
            pass

        pos, neg, neu = analyze_sentiment(headline)
        now_utc = datetime.now(timezone.utc)

        if not mercato_aperto:
            file_storico = "storico_notturno_bot.csv"
            file_esiste = os.path.isfile(file_storico)
            try:
                with open(file_storico, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_esiste:
                        writer.writerow(["timestamp", "symbol", "headline", "finbert_pos", "finbert_neg", "finbert_neu"])
                    writer.writerow([now_utc.isoformat(), symbol, headline, pos, neg, neu])
                print(" 💾 Mercato chiuso. Notizia archiviata per addestramento futuro.")
            except Exception as e:
                print(f" ⚠️ Errore nel salvataggio della notizia: {e}")
            continue

        current_price, vol = get_market_data(symbol)
        if current_price is None:
            continue

        features = pd.DataFrame([{
            "finbert_pos": pos,
            "finbert_neg": neg,
            "finbert_neu": neu,
            "bar_range_pct": vol,
            "hour_utc": now_utc.hour,
            "dow": now_utc.weekday(),
            "text_len": len(headline)
        }])

        try:
            probs = xgb_model.predict_proba(features)[0]
            prob_short, prob_hold, prob_long = probs[0], probs[1], probs[2]

            print(f" 📊 Probabilità Modello -> SHORT: {prob_short*100:.1f}% | HOLD: {prob_hold*100:.1f}% | LONG: {prob_long*100:.1f}%")

            if prob_long > SOGLIA_COMPRA:
                execute_bracket_trade(symbol, OrderSide.BUY, prob_long * 100, current_price, vol)
            elif prob_short > SOGLIA_VENDI:
                execute_bracket_trade(symbol, OrderSide.SELL, prob_short * 100, current_price, vol)
            else:
                print(" ⏳ Nessun segnale sufficientemente forte. Non opero.")
        except Exception as e:
            print(f" ⚠️ Errore durante la predizione o l'invio dell'ordine: {e}")


# ==========================================
# 7. AVVIO BOT
# ==========================================
async def main():
    asyncio.create_task(close_old_positions())
    news_stream.subscribe_news(on_news, "*")
    print("📡 Connessione al server WebSocket di Alpaca in corso...")
    await asyncio.to_thread(news_stream.run)


if __name__ == "__main__":
    asyncio.run(main())