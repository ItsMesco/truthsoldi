import os
import json
import argparse
import pandas as pd


DEFAULT_ROWS = [
    ["AAPL", "Apple Inc.", "technology", "consumer electronics", "stock"],
    ["MSFT", "Microsoft Corporation", "technology", "software", "stock"],
    ["NVDA", "NVIDIA Corporation", "technology", "semiconductors", "stock"],
    ["AMZN", "Amazon.com, Inc.", "consumer discretionary", "internet retail", "stock"],
    ["META", "Meta Platforms, Inc.", "communication services", "internet content", "stock"],
    ["GOOGL", "Alphabet Inc.", "communication services", "internet content", "stock"],
    ["TSLA", "Tesla, Inc.", "consumer discretionary", "auto manufacturers", "stock"],
    ["JPM", "JPMorgan Chase & Co.", "financials", "banks", "stock"],
    ["BAC", "Bank of America Corporation", "financials", "banks", "stock"],
    ["WFC", "Wells Fargo & Company", "financials", "banks", "stock"],
    ["XOM", "Exxon Mobil Corporation", "energy", "oil & gas", "stock"],
    ["CVX", "Chevron Corporation", "energy", "oil & gas", "stock"],
    ["LMT", "Lockheed Martin Corporation", "industrials", "defense", "stock"],
    ["NOC", "Northrop Grumman Corporation", "industrials", "defense", "stock"],
    ["RTX", "RTX Corporation", "industrials", "defense", "stock"],
    ["WMT", "Walmart Inc.", "consumer staples", "discount stores", "stock"],
    ["COST", "Costco Wholesale Corporation", "consumer staples", "discount stores", "stock"],
    ["UNH", "UnitedHealth Group Incorporated", "healthcare", "managed care", "stock"],
    ["PFE", "Pfizer Inc.", "healthcare", "drug manufacturers", "stock"],
    ["XLF", "Financial Select Sector SPDR Fund", "financials", "sector etf", "etf"],
    ["XLK", "Technology Select Sector SPDR Fund", "technology", "sector etf", "etf"],
    ["XLE", "Energy Select Sector SPDR Fund", "energy", "sector etf", "etf"],
    ["XLV", "Health Care Select Sector SPDR Fund", "healthcare", "sector etf", "etf"],
    ["XLI", "Industrial Select Sector SPDR Fund", "industrials", "sector etf", "etf"],
    ["XLY", "Consumer Discretionary Select Sector SPDR Fund", "consumer discretionary", "sector etf", "etf"],
    ["XLP", "Consumer Staples Select Sector SPDR Fund", "consumer staples", "sector etf", "etf"],
    ["QQQ", "Invesco QQQ Trust", "technology", "index etf", "etf"],
    ["SPY", "SPDR S&P 500 ETF Trust", "broad market", "index etf", "etf"],
    ["DIA", "SPDR Dow Jones Industrial Average ETF Trust", "broad market", "index etf", "etf"],
    ["IWM", "iShares Russell 2000 ETF", "small caps", "index etf", "etf"],
    ["TLT", "iShares 20+ Year Treasury Bond ETF", "bonds", "treasury etf", "bond_etf"],
    ["IEF", "iShares 7-10 Year Treasury Bond ETF", "bonds", "treasury etf", "bond_etf"],
    ["LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF", "bonds", "corporate bond etf", "bond_etf"],
    ["HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "bonds", "high yield bond etf", "bond_etf"],
    ["GLD", "SPDR Gold Shares", "gold", "commodity etf", "commodity_etf"],
    ["USO", "United States Oil Fund", "oil", "commodity etf", "commodity_etf"],
    ["UUP", "Invesco DB US Dollar Index Bullish Fund", "dollar", "currency etf", "etf"],
    ["VIXY", "ProShares VIX Short-Term Futures ETF", "volatility", "volatility etf", "etf"],
    ["FXI", "iShares China Large-Cap ETF", "china", "country etf", "etf"],
    ["SMH", "VanEck Semiconductor ETF", "technology", "semiconductors", "etf"],
    ["SOXX", "iShares Semiconductor ETF", "technology", "semiconductors", "etf"],
    ["XRT", "SPDR S&P Retail ETF", "consumer discretionary", "retail etf", "etf"],
    ["IBB", "iShares Biotechnology ETF", "healthcare", "biotech etf", "etf"],
]


def main(args):
    df = pd.DataFrame(DEFAULT_ROWS, columns=["symbol", "name", "sector", "industry", "asset_type"])
    if args.extra_csv and os.path.exists(args.extra_csv):
        extra = pd.read_csv(args.extra_csv)
        for col in ["symbol", "name"]:
            if col not in extra.columns:
                raise ValueError(f"Manca colonna {col} nel file extra")
        if "sector" not in extra.columns:
            extra["sector"] = "unknown"
        if "industry" not in extra.columns:
            extra["industry"] = "unknown"
        if "asset_type" not in extra.columns:
            extra["asset_type"] = "stock"
        df = pd.concat([df, extra[["symbol", "name", "sector", "industry", "asset_type"]]], ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "symbol_master.csv")
    df.to_csv(out, index=False)
    print(json.dumps({"rows": int(len(df)), "output": out}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-csv", default="")
    main(ap.parse_args())
