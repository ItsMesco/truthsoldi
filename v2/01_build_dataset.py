import os
import re
import json
import math
import html
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


HORIZONS_MIN = [5, 15, 30, 60]
VOL_Z_WINDOW = 20


def normalize_text(x: str) -> str:
    x = html.unescape(str(x or ""))
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"https?://\\S+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def combine_text(row: pd.Series) -> str:
    title = normalize_text(row.get("title", ""))
    summary = normalize_text(row.get("summary", ""))
    return (title + " [SEP] " + summary).strip()


def infer_event_type(text: str) -> str:
    t = text.lower()
    rules = [
        ("earnings", ["earnings", "eps", "revenue", "guidance", "quarter results", "q1", "q2", "q3", "q4"]),
        ("mna", ["acquire", "acquisition", "merger", "buyout", "takeover"]),
        ("executive", ["ceo", "cfo", "chairman", "resign", "resigns", "appointed", "fired", "board"]),
        ("scandal", ["probe", "investigation", "fraud", "bribery", "accounting", "sec investigation", "lawsuit", "whistleblower"]),
        ("analyst", ["upgrade", "downgrade", "price target", "overweight", "underweight", "buy rating", "sell rating"]),
        ("macro", ["cpi", "ppi", "payrolls", "fomc", "fed", "ecb", "rates", "inflation", "treasury", "gdp"]),
        ("commodity", ["oil", "crude", "gold", "copper", "natural gas", "opec"]),
        ("geopolitics", ["tariff", "sanction", "war", "china", "iran", "russia", "trade restrictions"]),
    ]
    for label, kws in rules:
        if any(k in t for k in kws):
            return label
    return "other"


def compute_forward_return(bars: pd.DataFrame, event_ts: pd.Timestamp, horizon_min: int) -> float:
    future_ts = event_ts + pd.Timedelta(minutes=horizon_min)
    sub = bars[bars["timestamp"] >= event_ts]
    if sub.empty:
        return np.nan
    p0 = sub.iloc[0]["close"]
    sub_f = bars[bars["timestamp"] >= future_ts]
    if sub_f.empty:
        return np.nan
    p1 = sub_f.iloc[0]["close"]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return float((p1 / p0) - 1.0)


def compute_premarket_gap(daily_prev_close: Optional[float], first_price: Optional[float]) -> float:
    if daily_prev_close in [None, 0] or first_price in [None, np.nan]:
        return np.nan
    return float((first_price / daily_prev_close) - 1.0)


def label_impact(ret: float, threshold: float) -> int:
    if pd.isna(ret):
        return -100
    if ret > threshold:
        return 2
    if ret < -threshold:
        return 0
    return 1


def add_market_features(news_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
    bars_df = bars_df.sort_values(["symbol", "timestamp"]).copy()
    bars_df["ret_1"] = bars_df.groupby("symbol")["close"].pct_change()
    bars_df["vol_mean_20"] = bars_df.groupby("symbol")["volume"].transform(lambda s: s.rolling(VOL_Z_WINDOW).mean())
    bars_df["vol_std_20"] = bars_df.groupby("symbol")["volume"].transform(lambda s: s.rolling(VOL_Z_WINDOW).std())
    bars_df["vol_z"] = (bars_df["volume"] - bars_df["vol_mean_20"]) / bars_df["vol_std_20"].replace(0, np.nan)
    bars_df["ret_abs_20_mean"] = bars_df.groupby("symbol")["ret_1"].transform(lambda s: s.abs().rolling(VOL_Z_WINDOW).mean())

    rows = []
    grouped = {k: g.reset_index(drop=True) for k, g in bars_df.groupby("symbol")}

    for _, row in news_df.iterrows():
        symbol = str(row.get("ticker", "")).strip().upper()
        if not symbol or symbol not in grouped:
            continue
        g = grouped[symbol]
        ts = row["published_utc"]
        sub = g[g["timestamp"] >= ts]
        if sub.empty:
            continue
        hit = sub.iloc[0]
        out = row.to_dict()
        out["event_price"] = float(hit["close"])
        out["event_volume"] = float(hit["volume"])
        out["vol_z"] = float(hit["vol_z"]) if pd.notna(hit["vol_z"]) else 0.0
        out["recent_abs_ret_mean"] = float(hit["ret_abs_20_mean"]) if pd.notna(hit["ret_abs_20_mean"]) else 0.0
        for h in HORIZONS_MIN:
            r = compute_forward_return(g, ts, h)
            out[f"fwd_ret_{h}m"] = r
        rows.append(out)

    return pd.DataFrame(rows)


def main(args):
    news = pd.read_csv(args.news_path)
    bars = pd.read_parquet(args.bars_path)

    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, errors="coerce")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")

    news = news.dropna(subset=["published_utc"]).copy()
    bars = bars.dropna(subset=["timestamp", "symbol", "close", "volume"]).copy()
    bars["symbol"] = bars["symbol"].astype(str).str.upper().str.strip()

    news["ticker"] = news.get("ticker", pd.Series([""] * len(news))).astype(str).str.upper().str.strip()
    news["source"] = news.get("source", pd.Series(["unknown"] * len(news))).astype(str)
    news["asset_type"] = news.get("asset_type", pd.Series(["stock"] * len(news))).astype(str)
    news["text"] = news.apply(combine_text, axis=1)
    news = news[news["text"].str.len() > 20].copy()
    news["event_type_weak"] = news["text"].map(infer_event_type)
    news["text_len"] = news["text"].str.len()
    news["hour_utc"] = news["published_utc"].dt.hour
    news["weekday"] = news["published_utc"].dt.weekday
    news["month"] = news["published_utc"].dt.month

    ds = add_market_features(news, bars)
    if ds.empty:
        raise RuntimeError("Dataset vuoto: controlla ticker/timestamp/news")

    thresholds = {5: args.th_5m, 15: args.th_15m, 30: args.th_30m, 60: args.th_60m}
    for h in HORIZONS_MIN:
        ds[f"y_ret_{h}m"] = ds[f"fwd_ret_{h}m"]
        ds[f"y_cls_{h}m"] = ds[f"fwd_ret_{h}m"].map(lambda x: label_impact(x, thresholds[h]))

    ds["severity_score_weak"] = (
        ds["vol_z"].clip(-5, 5).fillna(0).abs() * 0.35
        + ds["recent_abs_ret_mean"].fillna(0).clip(0, 0.05) * 15.0
        + ds["text_len"].clip(0, 500) / 500 * 0.15
    ).clip(0, 1)

    ds = ds[(ds[[f"y_cls_{h}m" for h in HORIZONS_MIN]] != -100).all(axis=1)].copy()

    label_maps = {
        "event_type": sorted(ds["event_type_weak"].dropna().unique().tolist()),
        "impact_class": {"0": "short", "1": "flat", "2": "long"},
        "asset_type": sorted(ds["asset_type"].dropna().unique().tolist()),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    ds.to_parquet(os.path.join(args.out_dir, "dataset_event_model.parquet"), index=False)
    with open(os.path.join(args.out_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)

    print({
        "rows": int(len(ds)),
        "symbols": int(ds["ticker"].nunique()),
        "event_types": ds["event_type_weak"].value_counts().to_dict(),
        "output": os.path.join(args.out_dir, "dataset_event_model.parquet"),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--news-path", required=True)
    ap.add_argument("--bars-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--th-5m", type=float, default=0.0025)
    ap.add_argument("--th-15m", type=float, default=0.0040)
    ap.add_argument("--th-30m", type=float, default=0.0060)
    ap.add_argument("--th-60m", type=float, default=0.0090)
    main(ap.parse_args())
