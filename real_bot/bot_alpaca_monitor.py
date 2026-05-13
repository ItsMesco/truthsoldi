import os
import re
import time
import json
import html
import pickle
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

import requests
import feedparser
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
QUANTITA_TRADE = int(os.getenv("TRADE_QTY", "1"))
CONFIDENZA_MINIMA = float(os.getenv("MIN_CONFIDENCE", "0.65"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "45"))
TRUTH_FEED_URL = os.getenv("TRUTH_FEED_URL", "https://trumpstruth.org/feed")
X_USER_ID = os.getenv("X_USER_ID", "25073877")

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
    return {"seen_ids": [], "seen_hashes": [], "last_orders": {}, "last_processed_at": None}


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
    key = f"{ticker}::{post_hash}"
    last_orders[key] = datetime.now(timezone.utc).isoformat()


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


def get_latest_x_post():
    if not X_BEARER_TOKEN:
        logger.warning("X_BEARER_TOKEN mancante: salto il controllo su X")
        return None
    url = f"https://api.x.com/2/users/{X_USER_ID}/tweets"
    params = {"max_results": 5, "exclude": "replies,retweets"}
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    tweet = data[0]
    text = tweet.get("text", "").strip()
    post_id = tweet.get("id")
    return {
        "source": "x",
        "id": f"x::{post_id}",
        "text": text,
        "published": None,
    }


def get_candidate_posts():
    posts = []
    try:
        p = get_latest_truth_post()
        if p:
            posts.append(p)
    except Exception as e:
        logger.warning(f"Errore Truth Social: {e}")
    try:
        p = get_latest_x_post()
        if p:
            posts.append(p)
    except Exception as e:
        logger.warning(f"Errore X: {e}")
    return posts


def analizza_e_trada_tutto(testo_post: str, state: dict):
    now = datetime.now(timezone.utc)
    post_hash = text_fingerprint(testo_post)
    logger.info(f"🚨 NUOVO POST RILEVATO | {testo_post}")

    try:
        req = StockLatestBarRequest(symbol_or_symbols=LISTA_TICKER)
        latest_bars = data_client.get_stock_latest_bar(req)
    except Exception as e:
        logger.error(f"Errore nel download dei prezzi live: {e}")
        return

    enc = tokenizer(testo_post, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    len_norm = float(scaler_len.transform([[len(testo_post)]])[0][0])
    etichette = {0: "SHORT", 1: "FLAT", 2: "LONG"}
    ordini_inviati = 0

    for ticker in LISTA_TICKER:
        if ticker not in latest_bars:
            continue

        close_live = latest_bars[ticker].close
        volume_live = latest_bars[ticker].volume
        v = scaler_vol.transform([[volume_live]])[0][0]
        c = scaler_close.transform([[close_live]])[0][0]
        t_idx = float(TICKER2IDX[ticker])

        feats = torch.tensor([[now.hour, now.minute, now.weekday(), now.month, v, c, len_norm, t_idx]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), feats)
            probs = torch.softmax(logits, dim=2).cpu().numpy()[0]

        prob_30m = probs[4]
        classe_predetta = int(np.argmax(prob_30m))
        confidenza = float(prob_30m[classe_predetta])
        decisione = etichette[classe_predetta]

        if decisione != "FLAT" and confidenza >= CONFIDENZA_MINIMA:
            if in_cooldown(state, ticker, post_hash):
                logger.info(f"Cooldown attivo su {ticker}: nessun ordine duplicato")
                continue

            logger.info(f"Segnale {ticker} | {decisione} | conf={confidenza:.1%} | prezzo=${close_live:.2f}")
            side = OrderSide.BUY if decisione == "LONG" else OrderSide.SELL
            try:
                ordine = MarketOrderRequest(symbol=ticker, qty=QUANTITA_TRADE, side=side, time_in_force=TimeInForce.GTC)
                trading_client.submit_order(order_data=ordine)
                ordini_inviati += 1
                mark_order(state, ticker, post_hash)
            except Exception as e:
                logger.error(f"Errore ordine {ticker}: {e}")

    if ordini_inviati == 0:
        logger.info("Nessun ticker ha superato la soglia o era in cooldown")
    else:
        logger.info(f"Operazione completata. Inviati {ordini_inviati} ordini.")


def monitor_posts():
    state = load_state()
    seen_ids = set(state.get("seen_ids", []))
    seen_hashes = set(state.get("seen_hashes", []))
    logger.info("👀 Monitor attivo: controllo periodico su Truth Social + X")

    while True:
        try:
            posts = get_candidate_posts()
            nuovi = []
            for post in posts:
                pid = post["id"]
                phash = text_fingerprint(post["text"])
                if pid in seen_ids:
                    continue
                if phash in seen_hashes:
                    continue
                nuovi.append((post, phash))

            if nuovi:
                post, phash = nuovi[0]
                logger.info(f"Nuovo post da {post['source'].upper()} | ID={post['id']}")
                analizza_e_trada_tutto(post["text"], state)
                seen_ids.add(post["id"])
                seen_hashes.add(phash)
                state["seen_ids"] = list(seen_ids)
                state["seen_hashes"] = list(seen_hashes)
                state["last_processed_at"] = datetime.now(timezone.utc).isoformat()
                save_state(state)
            else:
                logger.info("Nessun nuovo post")
        except Exception as e:
            logger.error(f"Errore nel loop principale: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    monitor_posts()
