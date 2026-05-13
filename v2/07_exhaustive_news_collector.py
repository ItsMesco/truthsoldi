import os
import re
import json
import time
import html
import hashlib
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import feedparser

HEADERS = {"User-Agent": "ItsMesco financial collector / contact local"}

EVENT_RULES = {
    "executive_resignation": ["ceo resign", "cfo resign", "steps down", "stepping down", "resigns", "resigned", "departure"],
    "executive_appointment": ["appoints ceo", "appoints cfo", "named ceo", "named cfo", "new ceo", "new cfo"],
    "earnings": ["earnings", "guidance", "eps", "revenue", "quarter", "q1", "q2", "q3", "q4"],
    "mna": ["acquire", "acquisition", "merger", "takeover", "buyout"],
    "scandal": ["probe", "investigation", "fraud", "lawsuit", "whistleblower", "sec investigation", "accounting"],
    "analyst": ["upgrade", "downgrade", "price target", "rating"],
    "macro": ["inflation", "cpi", "ppi", "fomc", "fed", "treasury", "rates", "payrolls"],
    "commodity": ["oil", "gold", "copper", "natural gas", "opec"],
    "ipo": ["ipo", "public offering", "new listing", "newly listed"],
    "buyback_dividend": ["buyback", "repurchase", "dividend", "special dividend"],
}

SECTOR_ETFS = {
    "technology": ["XLK", "QQQ", "SMH", "SOXX"],
    "financials": ["XLF", "KBE", "JPM", "BAC", "WFC"],
    "energy": ["XLE", "USO", "XOM", "CVX"],
    "healthcare": ["XLV", "IBB", "UNH", "PFE"],
    "industrials": ["XLI", "ITA", "LMT", "NOC", "RTX"],
    "consumer discretionary": ["XLY", "XRT", "AMZN", "TSLA", "COST"],
    "consumer staples": ["XLP", "WMT"],
    "materials": ["XLB", "SLX"],
    "utilities": ["XLU"],
    "communication services": ["XLC", "META", "GOOGL"],
    "bonds": ["TLT", "IEF", "LQD", "HYG"],
    "gold": ["GLD", "GDX"],
    "dollar": ["UUP"],
    "volatility": ["VIXY"],
    "broad market": ["SPY", "DIA", "IWM", "QQQ"],
    "china": ["FXI", "MCHI"],
}

DEFAULT_SYMBOL_MASTER = [
    ("AAPL", "Apple Inc.", "technology", "consumer electronics", "stock"),
    ("MSFT", "Microsoft Corporation", "technology", "software", "stock"),
    ("NVDA", "NVIDIA Corporation", "technology", "semiconductors", "stock"),
    ("AMZN", "Amazon.com, Inc.", "consumer discretionary", "internet retail", "stock"),
    ("META", "Meta Platforms, Inc.", "communication services", "internet content", "stock"),
    ("GOOGL", "Alphabet Inc.", "communication services", "internet content", "stock"),
    ("TSLA", "Tesla, Inc.", "consumer discretionary", "auto manufacturers", "stock"),
    ("JPM", "JPMorgan Chase & Co.", "financials", "banks", "stock"),
    ("BAC", "Bank of America Corporation", "financials", "banks", "stock"),
    ("WFC", "Wells Fargo & Company", "financials", "banks", "stock"),
    ("XOM", "Exxon Mobil Corporation", "energy", "oil & gas", "stock"),
    ("CVX", "Chevron Corporation", "energy", "oil & gas", "stock"),
    ("LMT", "Lockheed Martin Corporation", "industrials", "defense", "stock"),
    ("NOC", "Northrop Grumman Corporation", "industrials", "defense", "stock"),
    ("RTX", "RTX Corporation", "industrials", "defense", "stock"),
    ("WMT", "Walmart Inc.", "consumer staples", "discount stores", "stock"),
    ("COST", "Costco Wholesale Corporation", "consumer staples", "discount stores", "stock"),
    ("UNH", "UnitedHealth Group Incorporated", "healthcare", "managed care", "stock"),
    ("PFE", "Pfizer Inc.", "healthcare", "drug manufacturers", "stock"),
    ("XLF", "Financial Select Sector SPDR Fund", "financials", "sector etf", "etf"),
    ("XLK", "Technology Select Sector SPDR Fund", "technology", "sector etf", "etf"),
    ("XLE", "Energy Select Sector SPDR Fund", "energy", "sector etf", "etf"),
    ("XLV", "Health Care Select Sector SPDR Fund", "healthcare", "sector etf", "etf"),
    ("XLI", "Industrial Select Sector SPDR Fund", "industrials", "sector etf", "etf"),
    ("XLY", "Consumer Discretionary Select Sector SPDR Fund", "consumer discretionary", "sector etf", "etf"),
    ("XLP", "Consumer Staples Select Sector SPDR Fund", "consumer staples", "sector etf", "etf"),
    ("QQQ", "Invesco QQQ Trust", "technology", "index etf", "etf"),
    ("SPY", "SPDR S&P 500 ETF Trust", "broad market", "index etf", "etf"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF Trust", "broad market", "index etf", "etf"),
    ("IWM", "iShares Russell 2000 ETF", "small caps", "index etf", "etf"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "bonds", "treasury etf", "bond_etf"),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", "bonds", "treasury etf", "bond_etf"),
    ("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF", "bonds", "corporate bond etf", "bond_etf"),
    ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "bonds", "high yield bond etf", "bond_etf"),
    ("GLD", "SPDR Gold Shares", "gold", "commodity etf", "commodity_etf"),
    ("USO", "United States Oil Fund", "oil", "commodity etf", "commodity_etf"),
    ("UUP", "Invesco DB US Dollar Index Bullish Fund", "dollar", "currency etf", "etf"),
    ("VIXY", "ProShares VIX Short-Term Futures ETF", "volatility", "volatility etf", "etf"),
]

def norm(x):
    x = html.unescape(str(x or ""))
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"https?://\S+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def fingerprint(*parts):
    s = "|".join(norm(p) for p in parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_master(path):
    if path and os.path.exists(path):
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    else:
        df = pd.DataFrame(DEFAULT_SYMBOL_MASTER, columns=["symbol", "name", "sector", "industry", "asset_type"])
    for c in ["symbol", "name", "sector", "industry", "asset_type"]:
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df.drop_duplicates("symbol").reset_index(drop=True)

def detect_event(text):
    t = text.lower()
    for label, kws in EVENT_RULES.items():
        if any(k in t for k in kws):
            return label
    return "other"

def fetch_rss(url):
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries[:500]:
        title = norm(e.get("title", ""))
        summary = norm(e.get("summary", "") or e.get("description", ""))
        published = e.get("published", "") or e.get("updated", "")
        link = e.get("link", "")
        rows.append({"published_utc": published, "source": url, "title": title, "summary": summary, "url": link, "source_id": fingerprint(url, title, published, link)})
    return rows

def fetch_alpha_vantage_news(api_key, tickers, limit_per_ticker=100):
    out = []
    for t in tickers:
        try:
            r = requests.get("https://www.alphavantage.co/query", params={"function": "NEWS_SENTIMENT", "tickers": t, "apikey": api_key, "sort": "LATEST"}, timeout=30)
            data = r.json()
            for item in data.get("feed", [])[:limit_per_ticker]:
                title = norm(item.get("title", ""))
                summary = norm(item.get("summary", ""))
                pub = item.get("time_published", "")
                ts = ""
                if pub and len(pub) >= 14:
                    ts = datetime.strptime(pub[:14], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
                out.append({"published_utc": ts or pub, "source": "alpha_vantage", "title": title, "summary": summary, "ticker": t, "asset_type": "stock", "url": "https://www.alphavantage.co", "source_id": fingerprint("av", t, title, pub)})
        except Exception:
            pass
        time.sleep(0.25)
    return out

def fetch_finnhub_company_news(api_key, symbols, date_from, date_to):
    out = []
    for t in symbols:
        try:
            r = requests.get("https://finnhub.io/api/v1/company-news", params={"symbol": t, "from": date_from, "to": date_to, "token": api_key}, timeout=30)
            data = r.json()
            if isinstance(data, list):
                for item in data[:200]:
                    title = norm(item.get("headline", ""))
                    summary = norm(item.get("summary", ""))
                    pub = item.get("datetime")
                    ts = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat() if pub else ""
                    out.append({"published_utc": ts, "source": "finnhub", "title": title, "summary": summary, "ticker": t, "asset_type": "stock", "url": item.get("url", ""), "source_id": fingerprint("fh", t, title, ts)})
        except Exception:
            pass
        time.sleep(0.15)
    return out

def fetch_sec_recent(cik_path, max_ciks=400, lookback_days=90):
    if not cik_path or not os.path.exists(cik_path):
        return []
    df = pd.read_csv(cik_path)
    cik_col = df.columns[0]
    ciks = df[cik_col].astype(str).tolist()[:max_ciks]
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out = []
    headers = {"User-Agent": "ItsMesco financial collector / contact local", "Accept-Encoding": "gzip, deflate"}
    for cik in ciks:
        cik10 = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            forms = data.get("filings", {}).get("recent", {})
            company = data.get("name", "")
            tickers = data.get("tickers", []) or []
            ticker = tickers[0] if tickers else ""
            for form, dt_s, acc, doc in zip(forms.get("form", []), forms.get("filingDate", []), forms.get("accessionNumber", []), forms.get("primaryDocument", [])):
                ts = pd.to_datetime(dt_s, utc=True, errors="coerce")
                if pd.isna(ts) or ts.to_pydatetime() < cutoff:
                    continue
                if form not in ["8-K", "10-Q", "10-K", "6-K", "20-F", "S-1", "S-3"]:
                    continue
                out.append({"published_utc": ts.isoformat(), "source": "sec_edgar", "title": f"SEC filing {form} - {company}", "summary": f"Recent SEC filing {form} for {company}", "ticker": ticker, "asset_type": "stock", "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}", "source_id": fingerprint("sec", cik10, acc)})
        except Exception:
            pass
        time.sleep(0.08)
    return out

def infer_symbols(text, master):
    valid = set(master.symbol.tolist())
    hits = set(re.findall(r"[A-Z]{1,5}", text.upper())) & valid
    lower = text.lower()
    for _, row in master.iterrows():
        name = str(row["name"]).lower()
        if name and len(name) >= 4 and name in lower:
            hits.add(row["symbol"])
    return sorted(hits)

def related_assets(primary, master):
    symbols = set(primary)
    for s in primary:
        row = master[master.symbol == s]
        if row.empty:
            continue
        sec = str(row.iloc[0].get("sector", "")).lower()
        ind = str(row.iloc[0].get("industry", "")).lower()
        if sec in SECTOR_ETFS:
            symbols.update(SECTOR_ETFS[sec])
        if "semiconductor" in ind:
            symbols.update(["SMH", "SOXX", "NVDA", "AMD", "AVGO", "QCOM", "MU"])
        if "bank" in ind:
            symbols.update(["XLF", "KBE", "JPM", "BAC", "WFC", "C", "GS", "MS"])
        if "oil" in ind or sec == "energy":
            symbols.update(["XLE", "USO", "XOM", "CVX", "OXY", "SLB"])
        if "defense" in ind:
            symbols.update(["ITA", "LMT", "NOC", "RTX", "GD", "HII"])
    symbols.update(SECTOR_ETFS["broad market"])
    symbols.update(["VIXY", "UUP", "TLT", "IEF", "LQD", "HYG", "GLD", "USO"])
    return sorted(symbols)

def build_graph(df, master):
    rows = []
    edges = []
    for _, r in df.iterrows():
        text = f"{r.get('title','')} {r.get('summary','')}"
        event_type = detect_event(text)
        primary = infer_symbols(text, master)
        rel = related_assets(primary, master)
        eid = r['source_id']
        rows.append({"event_id": eid, "published_utc": r.get("published_utc", ""), "source": r.get("source", ""), "title": r.get("title", ""), "summary": r.get("summary", ""), "event_type": event_type, "primary_symbols": json.dumps(primary), "related_symbols": json.dumps(rel), "ticker": r.get("ticker", ""), "asset_type": r.get("asset_type", ""), "url": r.get("url", "")})
        for s in primary:
            edges.append({"event_id": eid, "src": eid, "dst": s, "edge_type": "PRIMARY", "weight": 1.0})
        for s in rel:
            if s not in primary:
                edges.append({"event_id": eid, "src": eid, "dst": s, "edge_type": "RELATED", "weight": 0.5})
    return pd.DataFrame(rows), pd.DataFrame(edges)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--news-out", default="raw_news.csv")
    ap.add_argument("--symbol-master", default="")
    ap.add_argument("--sec-cik-path", default="")
    ap.add_argument("--lookback-days", type=int, default=90)
    ap.add_argument("--max-ciks", type=int, default=400)
    ap.add_argument("--rss", nargs='*', default=["https://www.reuters.com/rssFeed/businessNews","https://www.cnbc.com/id/100003114/device/rss/rss.html","https://www.marketwatch.com/rss/topstories"])
    ap.add_argument("--archive-paths", nargs='*', default=[])
    ap.add_argument("--archive-format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--alpha-key", default=os.getenv("ALPHAVANTAGE_API_KEY", ""))
    ap.add_argument("--finnhub-key", default=os.getenv("FINNHUB_API_KEY", ""))
    ap.add_argument("--alpha-tickers", nargs='*', default=["AAPL","MSFT","NVDA","AMZN","TSLA","JPM","XOM","QQQ","SPY","TLT","GLD","USO","IWM","VIXY"])
    ap.add_argument("--finnhub-tickers", nargs='*', default=["AAPL","MSFT","NVDA","AMZN","TSLA","JPM","XOM","QQQ","SPY","TLT","GLD","USO","IWM"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    master = load_master(args.symbol_master)
    rows = []
    for u in args.rss:
        rows.extend(fetch_rss(u))
    if args.alpha_key:
        rows.extend(fetch_alpha_vantage_news(args.alpha_key, args.alpha_tickers))
    if args.finnhub_key:
        today = datetime.now(timezone.utc).date()
        rows.extend(fetch_finnhub_company_news(args.finnhub_key, args.finnhub_tickers, (today - timedelta(days=args.lookback_days)).isoformat(), today.isoformat()))
    rows.extend(fetch_sec_recent(args.sec_cik_path, args.max_ciks, args.lookback_days))
    for apath in args.archive_paths:
        if not apath or not os.path.exists(apath):
            continue
        try:
            if apath.lower().endswith(".parquet") or args.archive_format == "parquet":
                adf = pd.read_parquet(apath)
            else:
                adf = pd.read_csv(apath)
            needed = ["published_utc", "title"]
            for c in needed:
                if c not in adf.columns:
                    continue
            if "summary" not in adf.columns:
                adf["summary"] = ""
            if "ticker" not in adf.columns:
                adf["ticker"] = ""
            if "asset_type" not in adf.columns:
                adf["asset_type"] = "stock"
            if "source" not in adf.columns:
                adf["source"] = Path(apath).stem
            if "url" not in adf.columns:
                adf["url"] = ""
            adf["source_id"] = adf.apply(lambda r: fingerprint(apath, r.get("published_utc",""), r.get("title",""), r.get("ticker",""), r.get("url","")), axis=1)
            rows.extend(adf[["published_utc","source","title","summary","ticker","asset_type","url","source_id"]].to_dict("records"))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('Nessuna news raccolta')
    for c in ['published_utc','source','title','summary','ticker','asset_type','url','source_id']:
        if c not in df.columns:
            df[c] = ''
    df['title'] = df['title'].map(norm)
    df['summary'] = df['summary'].map(norm)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df['text'] = (df['title'] + ' [SEP] ' + df['summary']).str.strip()
    df['event_type_hint'] = df['text'].map(detect_event)
    df['primary_symbols'] = df['text'].map(lambda t: json.dumps(infer_symbols(t, master)))
    df['related_symbols'] = df['primary_symbols'].map(lambda s: json.dumps(related_assets(json.loads(s), master)))
    df = df.sort_values('published_utc').drop_duplicates(subset=['source_id']).reset_index(drop=True)
    graph_events, graph_edges = build_graph(df, master)
    df.to_csv(os.path.join(args.out_dir, args.news_out), index=False)
    graph_events.to_parquet(os.path.join(args.out_dir, 'news_graph_events.parquet'), index=False)
    graph_edges.to_parquet(os.path.join(args.out_dir, 'news_graph_edges.parquet'), index=False)
    print(json.dumps({'news_rows': int(len(df)), 'events': int(len(graph_events)), 'edges': int(len(graph_edges)), 'output': os.path.join(args.out_dir, args.news_out)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
