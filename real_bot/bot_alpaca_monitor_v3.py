import os
import re
import time
import json
import html
import pickle
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta

import feedparser
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
QUANTITA_TRADE = int(os.getenv("TRADE_QTY", "1"))
CONFIDENZA_LONG = float(os.getenv("MIN_CONFIDENCE_LONG", "0.70"))
CONFIDENZA_SHORT = float(os.getenv("MIN_CONFIDENCE_SHORT", "0.78"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "45"))
TRUTH_FEED_URL = os.getenv("TRUTH_FEED_URL", "https://trumpstruth.org/feed")
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
TIME_EXIT_MINUTES = int(os.getenv("TIME_EXIT_MINUTES", "90"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "5"))

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Mancano ALPACA_API_KEY o ALPACA_SECRET_KEY nel file .env")

TICKER2IDX = {
    'AAPL': 0, 'AMZN': 1, 'CAT': 2, 'DIA': 3, 'DJT': 4, 'DWAC': 5, 'EEM': 6,
    'FXI': 7, 'GLD': 8, 'ITA': 9, 'IWM': 10, 'JPM': 11, 'LMT': 12, 'NOC': 13,
    'NVDA': 14, 'QQQ': 15, 'SLX': 16, 'SPY': 17, 'TLT': 18, 'USO': 19, 'UUP': 20,
    'VIXY': 21, 'WMT': 22, 'XLE': 23, 'XLF': 24, 'XLI': 25, 'XLK': 26, 'XOM': 27, 'XRT': 28
}
LISTA_TICKER = list(TICKER2IDX.keys())

logger = logging.getLogger("trump_bot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(fh)
logger.addHandler(sh)

class HybridFinBERT(nn.Module):
    def __init__(self, n_numeric=8, n_targets=5, n_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")
        self.tabular = nn.Sequential(
            nn.Linear(n_numeric, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.heads = nn.ModuleList([nn.Linear(128, n_classes) for _ in range(n_targets)])

    def forward(self, input_ids, attention_mask, features):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        tab = self.tabular(features)
        fused = self.fusion(torch.cat([cls, tab], dim=1))
        return torch.stack([head(fused) for head in self.heads], dim=1)


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seen_ids": [], "seen_hashes": [], "last_orders": {}, "open_trades": {}, "last_processed_at": None}


def save_state(state):
    state["seen_ids"] = state.get("seen_ids", [])[-500:]
    state["seen_hashes"] = state.get("seen_hashes", [])[-500:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def parse_dt(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def in_cooldown(state, ticker: str, post_hash: str) -> bool:
    last_orders = state.get("last_orders", {})
    key = f"{ticker}::{post_hash}"
    ts = last_orders.get(key)
    if not ts:
        return False
    last_dt = parse_dt(ts)
    if not last_dt:
        return False
    delta_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0
    return delta_min < COOLDOWN_MINUTES


def mark_order(state, ticker: str, post_hash: str):
    last_orders = state.setdefault("last_orders", {})
    last_orders[f"{ticker}::{post_hash}"] = datetime.now(timezone.utc).isoformat()


def scale_single_feature(scaler, value, feature_name: str) -> float:
    df = pd.DataFrame([{feature_name: value}])
    return float(scaler.transform(df)[0][0])


def print_prediction_block(testo_post: str, ticker: str, now: datetime, probs: np.ndarray):
    horizons = ['1m ', '5m ', '10m', '15m', '30m']
    etichette = {0: '🔴 SHORT', 1: '🟡 FLAT ', 2: '🟢 LONG '}
    print(f"\n{'='*58}")
    print(f"📢 \"{testo_post[:65]}{'...' if len(testo_post)>65 else ''}\"")
    print(f"📈 Ticker: ${ticker} | 🕐 {now.strftime('%H:%M UTC')}")
    print(f"{'='*58}")
    print(f"  {'Orizzonte':<6} {'SHORT':>7} {'FLAT':>7} {'LONG':>7}   Decisione")
    print(f"  {'─'*50}")
    decisioni = []
    for h, p in zip(horizons, probs):
        p_short, p_flat, p_long = p
        decisione = etichette[int(np.argmax(p))]
        decisioni.append(int(np.argmax(p)))
        print(f"  {h:<6} {p_short:>6.0%}  {p_flat:>6.0%}  {p_long:>6.0%}   {decisione}")
    long_count = decisioni.count(2)
    short_count = decisioni.count(0)
    print(f"\n  {'─'*50}")
    if long_count >= 3:
        print(f"  🚀 ENTRA LONG   ({long_count}/5 orizzonti positivi)")
    elif short_count >= 3:
        print(f"  🔻 ENTRA SHORT  ({short_count}/5 orizzonti negativi)")
    elif long_count >= 2 and decisioni[0] == 2:
        print(f"  ⚡ SCALPING LONG breve (1m-5m positivi, poi esce)")
    else:
        print(f"  🛑 STAI FUORI   (segnale misto o piatto)")
    print(f"{'='*58}\n")


def count_open_positions(trading_client):
    try:
        positions = trading_client.get_all_positions()
        return len(positions)
    except Exception as e:
        logger.warning(f"Impossibile leggere le posizioni aperte: {e}")
        return 999


def get_latest_truth_post():
    feed = feedparser.parse(TRUTH_FEED_URL)
    if not feed.entries:
        return None
    e = feed.entries[0]
    text = e.get("title", "") or e.get("summary", "") or ""
    post_id = e.get("id") or e.get("link") or f"truth_{text_fingerprint(text)}"
    published = e.get("published", "")
    return {
        "source": "truth",
        "id": f"truth::{post_id}",
        "text": html.unescape(text).strip(),
        "published": published,
    }


logger.info("⏳ Caricamento risorse AI...")
with open("finbert_trading_bot/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
scaler_vol = scalers["volume"]
scaler_close = scalers["close"]
scaler_len = scalers["len"]

tokenizer = AutoTokenizer.from_pretrained("finbert_trading_bot/tokenizer")
model = HybridFinBERT(n_numeric=8).to(DEVICE)
model.load_state_dict(torch.load("finbert_trading_bot/model_weights.pt", map_location=DEVICE))
model.eval()

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
account = trading_client.get_account()
logger.info(f"✅ Connesso ad Alpaca Paper | Buying Power: ${account.buying_power}")
logger.info("ℹ️ Strategia attiva: ingresso selettivo + bracket order + time exit")


def cleanup_closed_trades(state: dict):
    open_trades = state.get("open_trades", {})
    if not open_trades:
        return
    try:
        current_positions = {p.symbol for p in trading_client.get_all_positions()}
    except Exception as e:
        logger.warning(f"Impossibile pulire open_trades: {e}")
        return
    to_delete = [ticker for ticker in open_trades if ticker not in current_positions]
    for ticker in to_delete:
        logger.info(f"Trade {ticker} non più aperto: rimosso dallo stato locale")
        del open_trades[ticker]


def enforce_time_exit(state: dict):
    open_trades = state.get("open_trades", {})
    if not open_trades:
        return
    now = datetime.now(timezone.utc)
    for ticker, meta in list(open_trades.items()):
        opened_at = parse_dt(meta.get("opened_at"))
        if not opened_at:
            continue
        age_min = (now - opened_at).total_seconds() / 60.0
        if age_min < TIME_EXIT_MINUTES:
            continue
        side = meta.get("side")
        qty = meta.get("qty", QUANTITA_TRADE)
        close_side = OrderSide.SELL if side == "LONG" else OrderSide.BUY
        try:
            logger.info(f"⏰ Time exit su {ticker}: posizione aperta da {age_min:.0f} min, chiusura forzata")
            ordine = MarketOrderRequest(symbol=ticker, qty=qty, side=close_side, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order_data=ordine)
            del open_trades[ticker]
        except Exception as e:
            logger.error(f"Errore chiusura time-exit {ticker}: {e}")


def decide_entry(probs: np.ndarray):
    p30 = probs[4]
    cls30 = int(np.argmax(p30))
    conf30 = float(p30[cls30])
    
    # Conta orizzonti concordi
    long_count = sum(1 for p in probs if int(np.argmax(p)) == 2)
    short_count = sum(1 for p in probs if int(np.argmax(p)) == 0)
    
    if cls30 == 2 and conf30 >= CONFIDENZA_LONG and long_count >= 2:
        return "LONG", conf30, long_count, short_count
    if cls30 == 0 and conf30 >= CONFIDENZA_SHORT and short_count >= 2:
        return "SHORT", conf30, long_count, short_count
    return "FLAT", conf30, long_count, short_count



def submit_bracket_order(ticker: str, side_label: str, qty: int, close_live: float):
    if side_label == "LONG":
        side = OrderSide.BUY
        tp_price = round(close_live * (1 + TAKE_PROFIT_PCT), 2)
        sl_price = round(close_live * (1 - STOP_LOSS_PCT), 2)
    else:
        side = OrderSide.SELL
        tp_price = round(close_live * (1 - TAKE_PROFIT_PCT), 2)
        sl_price = round(close_live * (1 + STOP_LOSS_PCT), 2)

    ordine = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
    )
    trading_client.submit_order(order_data=ordine)
    return tp_price, sl_price


def analizza_e_trada_tutto(testo_post: str, state: dict):
    now = datetime.now(timezone.utc)
    post_hash = text_fingerprint(testo_post)
    logger.info(f"🚨 NUOVO POST RILEVATO | {testo_post}")

    cleanup_closed_trades(state)
    enforce_time_exit(state)

    try:
        open_positions = count_open_positions(trading_client)
        if open_positions >= MAX_OPEN_POSITIONS:
            logger.warning(f"Troppe posizioni aperte ({open_positions}). Nessun nuovo ingresso.")
            return
    except Exception:
        pass

    try:
        req = StockLatestBarRequest(symbol_or_symbols=LISTA_TICKER)
        latest_bars = data_client.get_stock_latest_bar(req)
    except Exception as e:
        logger.error(f"Errore nel download dei prezzi live: {e}")
        return

    enc = tokenizer(testo_post, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    len_norm = scale_single_feature(scaler_len, len(testo_post), "len")
    ordini_inviati = 0

    for ticker in LISTA_TICKER:
        if ticker not in latest_bars:
            continue
        if ticker in state.get("open_trades", {}):
            logger.info(f"{ticker} | posizione già aperta, salto nuovo ingresso")
            continue

        close_live = latest_bars[ticker].close
        volume_live = latest_bars[ticker].volume
        v = scale_single_feature(scaler_vol, volume_live, "volume")
        c = scale_single_feature(scaler_close, close_live, "close")
        t_idx = float(TICKER2IDX[ticker])
        feats = torch.tensor([[now.hour, now.minute, now.weekday(), now.month, v, c, len_norm, t_idx]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE), feats)
            probs = torch.softmax(logits, dim=2).cpu().numpy()[0]

        print_prediction_block(testo_post, ticker, now, probs)
        decisione, conf30, long_count, short_count = decide_entry(probs)

        if decisione == "FLAT":
            logger.info(f"{ticker} | Nessun movimento | conf30={conf30:.1%} | long_count={long_count} | short_count={short_count}")
            continue

        if in_cooldown(state, ticker, post_hash):
            logger.info(f"{ticker} | cooldown attivo, nessun ordine duplicato")
            continue

        try:
            tp_price, sl_price = submit_bracket_order(ticker, decisione, QUANTITA_TRADE, close_live)
            ordini_inviati += 1
            mark_order(state, ticker, post_hash)
            state.setdefault("open_trades", {})[ticker] = {
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "side": decisione,
                "qty": QUANTITA_TRADE,
                "entry_ref_price": round(close_live, 2),
                "tp_price": tp_price,
                "sl_price": sl_price,
                "post_hash": post_hash,
            }
            logger.info(
                f"✅ {ticker} | Entrata {decisione} | conf30={conf30:.1%} | entry≈${close_live:.2f} | TP=${tp_price:.2f} | SL=${sl_price:.2f}"
            )
        except Exception as e:
            logger.error(f"Errore ordine bracket {ticker}: {e}")

    if ordini_inviati == 0:
        logger.info("🛑 Nessun ticker ha soddisfatto i criteri di ingresso. Nessun trade eseguito.")
    else:
        logger.info(f"✅ Operazione completata. Inviati {ordini_inviati} ordini bracket.")


def monitor_posts():
    state = load_state()
    seen_ids = set(state.get("seen_ids", []))
    seen_hashes = set(state.get("seen_hashes", []))
    logger.info("👀 Monitor attivo: controllo periodico su Truth RSS")

    while True:
        try:
            cleanup_closed_trades(state)
            enforce_time_exit(state)
            post = get_latest_truth_post()
            if post is not None:
                pid = post["id"]
                phash = text_fingerprint(post["text"])
                if pid not in seen_ids and phash not in seen_hashes:
                    logger.info(f"Nuovo post da TRUTH | ID={pid}")
                    analizza_e_trada_tutto(post["text"], state)
                    seen_ids.add(pid)
                    seen_hashes.add(phash)
                    state["seen_ids"] = list(seen_ids)
                    state["seen_hashes"] = list(seen_hashes)
                    state["last_processed_at"] = datetime.now(timezone.utc).isoformat()
                    save_state(state)
                else:
                    logger.info("Nessun nuovo post")
            else:
                logger.info("Feed Truth vuoto o non disponibile")
        except Exception as e:
            logger.error(f"Errore nel loop principale: {e}")
        save_state(state)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    monitor_posts()
