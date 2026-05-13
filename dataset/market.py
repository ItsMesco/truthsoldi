import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf
import time

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================


API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") 
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Data spartiacque: prima → yfinance 5min, dopo → Alpaca 1min
CUTOFF = pd.Timestamp("2016-01-01", tz="UTC")

ALL_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",          # Benchmark
    "VIXY",                                # Volatilità (dal 2011)
    "XLK", "NVDA", "AAPL",               # Tech
    "ITA", "LMT", "NOC",                 # Difesa
    "XLE", "XOM", "USO",                 # Energia
    "XLF", "JPM",                         # Finanza
    "XLI", "SLX", "CAT",                 # Industria
    "XRT", "WMT", "AMZN",               # Retail
    "GLD", "TLT", "UUP",                 # Safe Haven
    "FXI", "EEM",                         # Cina / EM
    "DWAC", "DJT",                        # Trump-specific
]

# Ticker con data di inizio reale (non esistevano prima)
TICKER_START = {
    "VIXY": pd.Timestamp("2011-01-04", tz="UTC"),
    "DWAC": pd.Timestamp("2021-09-03", tz="UTC"),
    "DJT":  pd.Timestamp("2024-03-26", tz="UTC"),
    "NVDA": pd.Timestamp("2009-01-02", tz="UTC"),
}

# ==========================================
# 2. CARICAMENTO POST
# ==========================================
print("📂 Caricamento post Trump...")
df_posts = pd.read_csv('tutti_i_post_trump_uniti.csv')
df_posts['timestamp'] = pd.to_datetime(df_posts['timestamp'], format='mixed', utc=True)
df_posts = df_posts.sort_values('timestamp').reset_index(drop=True)

data_inizio = df_posts['timestamp'].min()
data_fine   = df_posts['timestamp'].max()
print(f"   Post totali: {len(df_posts):,}")
print(f"   Periodo:     {data_inizio.date()} → {data_fine.date()}")

# ==========================================
# 3A. DOWNLOAD PRE-2016 CON YFINANCE (5 min)
#     yfinance limita a ~60 giorni per chunk → loop automatico
# ==========================================
def scarica_yfinance_chunked(ticker, start, end, interval="5m"):
    """
    yfinance NON ha dati intraday storici oltre i 60 giorni.
    Strategia a cascata:
      - Prova prima con 1h (disponibile ~2 anni indietro)
      - Se fallisce, usa 1d (disponibile dal 1993, senza limiti)
    Per post del 2009-2015 il dato giornaliero è sufficiente:
    Trump twittava raramente e l'impatto si misurava in ore, non minuti.
    """
    end_cap = min(end, CUTOFF).replace(tzinfo=None)
    start_naive = start.replace(tzinfo=None)

    # ── Tenta 1h (funziona per gli ultimi ~2 anni) ──────────
    print(f"      ⏳ Provo 1h...")
    try:
        df = yf.download(
            ticker,
            start=start_naive,
            end=end_cap,
            interval="1h",
            progress=False,
            auto_adjust=True
        )
        if not df.empty and len(df) > 10:
            df = _normalizza_yf(df, ticker, '1h')
            print(f"      ✅ 1h OK → {len(df):,} candele")
            return df
    except Exception:
        pass

    # ── Fallback: 1d (disponibile dal 1993, nessun limite) ──
    print(f"      ⏳ Fallback su 1d (dati giornalieri)...")
    try:
        df = yf.download(
            ticker,
            start=start_naive,
            end=end_cap,
            interval="1d",
            progress=False,
            auto_adjust=True
        )
        if not df.empty:
            df = _normalizza_yf(df, ticker, '1d')
            print(f"      ✅ 1d OK → {len(df):,} candele")
            return df
    except Exception as e:
        print(f"      ❌ Anche 1d fallito: {e}")

    return pd.DataFrame()


def _normalizza_yf(df, ticker, granularity):
    """Pulisce e standardizza l'output di yfinance."""
    df = df.reset_index()

    # yfinance recente usa MultiIndex sulle colonne → appiattisci
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' or col[1] == ticker
                      else f"{col[0]}_{col[1]}"
                      for col in df.columns]

    # Rinomina colonna temporale (può chiamarsi 'Date' o 'Datetime')
    for possible in ['Datetime', 'Date', 'index']:
        if possible in df.columns:
            df = df.rename(columns={possible: 'timestamp'})
            break

    df = df.rename(columns={'Open': 'open', 'Close': 'close', 'Volume': 'volume'})
    df['timestamp']   = pd.to_datetime(df['timestamp'], utc=True)
    df['symbol']      = ticker
    df['granularity'] = granularity

    return df[['timestamp', 'symbol', 'open', 'close', 'volume', 'granularity']]


# ==========================================
# 3B. DOWNLOAD POST-2016 CON ALPACA (1 min)
# ==========================================
def scarica_alpaca(ticker, start, end):
    """Scarica candele a 1 minuto da Alpaca."""
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Minute,
            start=max(start, CUTOFF),
            end=end
        )
        bars = client.get_stock_bars(req)
        df = bars.df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['granularity'] = '1min'
        return df[['timestamp', 'symbol', 'open', 'close', 'volume', 'granularity']]
    except Exception as e:
        print(f"      ❌ Alpaca error: {e}")
        return pd.DataFrame()


# ==========================================
# 4. DOWNLOAD COMPLETO PER TUTTI I TICKER
# ==========================================
print("\n⏳ Download dati storici (fonte doppia: yfinance + Alpaca)...\n")
tutti_i_bar = []

for ticker in ALL_TICKERS:
    ticker_start = TICKER_START.get(ticker, data_inizio)

    # — Parte 1: pre-2016 da yfinance (solo se il ticker esisteva prima)
    if ticker_start < CUTOFF:
        print(f"   📥 {ticker:<6} [yfinance 5min]  "
              f"{ticker_start.date()} → {CUTOFF.date()}...")
        df_yf = scarica_yfinance_chunked(ticker, ticker_start, CUTOFF)
        if not df_yf.empty:
            tutti_i_bar.append(df_yf)
            print(f"          ✅ {len(df_yf):,} candele")
        else:
            print(f"          ⚠️  Nessun dato yfinance")

    # — Parte 2: post-2016 da Alpaca (solo se il ticker esisteva dopo)
    alpaca_start = max(ticker_start, CUTOFF)
    if alpaca_start < data_fine:
        print(f"   📥 {ticker:<6} [Alpaca  1min]   "
              f"{alpaca_start.date()} → {data_fine.date()}...")
        df_alp = scarica_alpaca(ticker, alpaca_start, data_fine)
        if not df_alp.empty:
            tutti_i_bar.append(df_alp)
            print(f"          ✅ {len(df_alp):,} candele")

df_mercato = pd.concat(tutti_i_bar, ignore_index=True)
print(f"\n✅ Download totale: {len(df_mercato):,} candele su {df_mercato['symbol'].nunique()} ticker")

# ==========================================
# 5. MERGE POST + PREZZI CON GRANULARITÀ ADATTIVA
# ==========================================
print("\n🔗 Merge post + prezzi per ogni ticker...")

intervalli_minuti = [1, 5, 10, 15, 30]
frammenti = []

# Soglie adattive per granularità
SOGLIE = {
    '1d':   1.0,
    '1h':   0.5,
    '5min': 0.3,
    '1min': 0.2,
}

# Tolleranze adattive per granularità
TOLERANCE = {
    '1d':   pd.Timedelta('2d'),
    '1h':   pd.Timedelta('6h'),
    '5min': pd.Timedelta('12h'),
    '1min': pd.Timedelta('12h'),
}

# ── Definita FUORI dal loop per evitare il bug della closure Python ──
def assegna_target(var, soglia):
    if var > soglia:
        return 2   # LONG  🟢
    elif var < -soglia:
        return 0   # SHORT 🔴
    else:
        return 1   # FLAT  🟡

for ticker in ALL_TICKERS:
    df_t = df_mercato[df_mercato['symbol'] == ticker][
        ['timestamp', 'open', 'close', 'volume', 'granularity']
    ].copy().sort_values('timestamp').reset_index(drop=True)

    if df_t.empty:
        print(f"   ⏭️  {ticker} — nessun dato, skip")
        continue

    frammenti_ticker = []

    for gran, df_gran in df_t.groupby('granularity'):
        soglia   = SOGLIE.get(gran, 0.2)
        toleranz = TOLERANCE.get(gran, pd.Timedelta('12h'))

        # Solo i post nel periodo coperto da questa granularità
        t_min = df_gran['timestamp'].min()
        t_max = df_gran['timestamp'].max()
        posts_periodo = df_posts[
            (df_posts['timestamp'] >= t_min) &
            (df_posts['timestamp'] <= t_max)
        ].copy()

        if posts_periodo.empty:
            continue

        merged = pd.merge_asof(
            posts_periodo,
            df_gran[['timestamp', 'open', 'close', 'volume']],
            on='timestamp',
            direction='forward',
            tolerance=toleranz
        )
        merged = merged.dropna(subset=['close'])

        if merged.empty:
            continue

        merged['ticker']      = ticker
        merged['granularity'] = gran

        # Calcola variazioni e target per ogni orizzonte temporale
        for m in intervalli_minuti:
            merged[f'Prezzo_Tra_{m}m'] = merged['close'].shift(-m)
            merged[f'Var_{m}m_%'] = (
                (merged[f'Prezzo_Tra_{m}m'] - merged['close']) / merged['close']
            ) * 100
            # ── Usa una variabile locale per catturare soglia correttamente ──
            soglia_locale = soglia
            merged[f'Target_{m}m'] = merged[f'Var_{m}m_%'].apply(
                lambda x, s=soglia_locale: assegna_target(x, s)
            )

        merged = merged.dropna()
        frammenti_ticker.append(merged)

    if frammenti_ticker:
        df_ticker_completo = pd.concat(frammenti_ticker, ignore_index=True)
        frammenti.append(df_ticker_completo)
        print(f"   ✅ {ticker:<6}  {len(df_ticker_completo):,} righe  "
              f"(granularità: {df_ticker_completo['granularity'].unique().tolist()})")

dataset_finale = pd.concat(frammenti, ignore_index=True)
print(f"\n✅ Merge completato: {len(dataset_finale):,} righe totali")

# ── Verifica distribuzione classi ──────────────────────────────────────
print("\n📊 Distribuzione classi (obiettivo: SHORT e LONG almeno 20% ciascuno):")
for m in intervalli_minuti:
    col = f'Target_{m}m'
    counts = dataset_finale[col].value_counts().sort_index()
    totale = len(dataset_finale)
    print(f"\n  Orizzonte {m}m:")
    print(f"    🔴 SHORT (0): {counts.get(0,0):>7,}  ({counts.get(0,0)/totale*100:.1f}%)")
    print(f"    🟡 FLAT  (1): {counts.get(1,0):>7,}  ({counts.get(1,0)/totale*100:.1f}%)")
    print(f"    🟢 LONG  (2): {counts.get(2,0):>7,}  ({counts.get(2,0)/totale*100:.1f}%)")

# ==========================================
# 6. FEATURE EXTRA + PERIODO PRESIDENZIALE
# ==========================================
dataset_finale['ora']               = dataset_finale['timestamp'].dt.hour
dataset_finale['giorno_settimana']  = dataset_finale['timestamp'].dt.dayofweek
dataset_finale['mese']              = dataset_finale['timestamp'].dt.month
dataset_finale['anno']              = dataset_finale['timestamp'].dt.year
dataset_finale['len_post']          = dataset_finale['testo'].str.len()

dataset_finale['periodo'] = 'fuori_mandato'
mask1 = (dataset_finale['anno'] >= 2017) & (dataset_finale['anno'] < 2021)
mask2 =  dataset_finale['anno'] >= 2025
dataset_finale.loc[mask1, 'periodo'] = 'mandato_1'
dataset_finale.loc[mask2, 'periodo'] = 'mandato_2'

# ==========================================
# 7. SALVATAGGIO
# ==========================================
dataset_finale.to_parquet(
    'dataset_IA_multi_ticker_completo.parquet',
    index=False,
    compression='snappy'   # Compressione veloce, ottima per ML
)
print(f"💾 Dimensione file: {os.path.getsize('dataset_IA_multi_ticker_completo.parquet') / 1e6:.1f} MB")

print(f"\n{'='*60}")
print(f"✅ DATASET COMPLETO PRONTO!")
print(f"   Righe totali:    {len(dataset_finale):,}")
print(f"   Post unici:      {dataset_finale['timestamp'].nunique():,}")
print(f"   Ticker coperti:  {dataset_finale['ticker'].nunique()}")
print(f"   Periodo:         {dataset_finale['timestamp'].min().date()} "
      f"→ {dataset_finale['timestamp'].max().date()}")
print(f"\n📊 Righe per ticker:")
print(dataset_finale.groupby('ticker')['timestamp'].count()
      .sort_values(ascending=False).to_string())
print(f"\n📅 Righe per granularità:")
print(dataset_finale.groupby('granularity')['timestamp'].count().to_string())
print(f"{'='*60}")





