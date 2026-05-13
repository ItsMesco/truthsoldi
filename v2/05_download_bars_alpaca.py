import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dateutil.relativedelta import relativedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def month_windows(start_dt, end_dt):
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + relativedelta(months=1), end_dt)
        yield cur, nxt
        cur = nxt


def get_timeframe(amount: int):
    return TimeFrame(amount, TimeFrameUnit.Minute)


def load_symbols(symbol_master_path: str, limit: int = 0):
    df = pd.read_csv(symbol_master_path)
    syms = df['symbol'].astype(str).str.upper().dropna().unique().tolist()
    if limit and limit > 0:
        syms = syms[:limit]
    return syms


def safe_get_bars(client, symbols, start_dt, end_dt, timeframe, feed=None, max_retries=2):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(symbol_or_symbols=symbols, timeframe=timeframe, start=start_dt, end=end_dt)
            if feed:
                kwargs['feed'] = feed
            req = StockBarsRequest(**kwargs)
            
            response = client.get_stock_bars(req)
            if response.df.empty:
                return pd.DataFrame()
                
            return response.df.reset_index()
            
        except Exception as e:
            last_err = e
            error_msg = str(e)
            
            # Stampa l'errore in rosso per il debug visivo
            print(f"\n    [!] Errore API Alpaca: {error_msg}")
            
            # Se l'errore è di autorizzazione o piano gratuito, interrompi i tentativi
            if any(code in error_msg for code in ['401', '403', 'Unauthorized', 'Forbidden', 'subscription']):
                return pd.DataFrame()
                
            time.sleep(0.5 * attempt)
            
    return pd.DataFrame()


def main(args):
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    if not key or not secret:
        raise RuntimeError('Mancano ALPACA_API_KEY / ALPACA_SECRET_KEY nelle variabili ambiente')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'bars_5m.parquet'

    symbols = load_symbols(args.symbol_master_path, args.limit)
    if not symbols:
        raise RuntimeError('Nessun simbolo trovato nel symbol master')

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    timeframe = get_timeframe(args.timeframe_amount)
    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    all_parts = []
    for batch_idx, batch_symbols in enumerate(chunks(symbols, args.batch_size), start=1):
        print(f'[batch {batch_idx}] symbols={len(batch_symbols)} first={batch_symbols[0]}')
        for w_start, w_end in month_windows(start_dt, end_dt):
            df = safe_get_bars(
                client,
                batch_symbols,
                w_start,
                w_end,
                timeframe,
                feed=args.feed if args.feed and args.feed.lower() != 'auto' else None,
            )
            if df is not None and not df.empty:
                all_parts.append(df)
                print(f'  window {w_start.date()} -> {w_end.date()} rows={len(df)}')
            else:
                print(f'  window {w_start.date()} -> {w_end.date()} skip')
            time.sleep(args.sleep_seconds)

    if not all_parts:
        raise RuntimeError('Nessuna barra scaricata da Alpaca. Controlla i log sopra per errori di autorizzazione (403/Subscription).')

    bars = pd.concat(all_parts, ignore_index=True)
    if 'timestamp' in bars.columns:
        bars['timestamp'] = pd.to_datetime(bars['timestamp'], utc=True, errors='coerce')
    bars = bars.drop_duplicates().sort_values([c for c in ['symbol', 'timestamp'] if c in bars.columns]).reset_index(drop=True)
    bars.to_parquet(out_file, index=False)
    print({'rows': int(len(bars)), 'symbols': int(bars['symbol'].nunique() if 'symbol' in bars.columns else 0), 'output': str(out_file)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol-master-path', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--timeframe-amount', type=int, default=5)
    ap.add_argument('--feed', default='auto')
    ap.add_argument('--batch-size', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sleep-seconds', type=float, default=0.1)
    main(ap.parse_args())