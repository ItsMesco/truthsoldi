import os
import sys
import json
import shlex
import argparse
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Manca pyyaml. Installa con: pip install pyyaml", file=sys.stderr)
    raise


def run(cmd, env=None):
    if env is None:
        env = os.environ.copy()
    print("\n>>>", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def load_env_file(env_path: str):
    if not env_path or not os.path.exists(env_path):
        return
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--mode", choices=["local_data", "local_news", "kaggle_train", "all_local"], default="local_data")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    load_env_file(args.env_file)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).resolve().parent
    py = args.python
    out_dir = cfg["project"]["output_dir_local"]
    os.makedirs(out_dir, exist_ok=True)

    symbol_master = cfg["local_paths"]["symbol_master"]
    sec_cik_list = cfg["local_paths"]["sec_cik_list"]
    raw_news = cfg["local_paths"]["raw_news"]
    bars_5m = cfg["local_paths"]["bars_5m"]
    dataset_path = cfg["local_paths"]["dataset_event_model"]
    model_artifacts = cfg["local_paths"]["model_artifacts"]

    collector = cfg["collector"]
    bars = cfg["bars_download"]
    train = cfg["training"]

    if args.mode in ["local_data", "local_news", "all_local"]:
        run([py, str(root / "06_build_symbol_master.py"), "--out-dir", out_dir])

        cmd_news = [
            py, str(root / "07_exhaustive_news_collector.py"),
            "--out-dir", out_dir,
            "--news-out", Path(raw_news).name,
            "--symbol-master", symbol_master,
            "--sec-cik-path", sec_cik_list,
            "--lookback-days", str(collector["lookback_days"]),
            "--max-ciks", str(collector["max_ciks"]),
            "--alpha-tickers", *collector["alpha_tickers"],
            "--finnhub-tickers", *collector["finnhub_tickers"],
            "--rss", *collector["rss"],
        ]
        if collector.get("archive_paths"):
            cmd_news += ["--archive-paths", *collector["archive_paths"]]
        run(cmd_news)

    if args.mode in ["local_data", "all_local"]:
        cmd_bars = [
            py, str(root / "05_download_bars_alpaca.py"),
            "--symbol-master-path", symbol_master,
            "--out-dir", out_dir,
            "--start", str(bars["start"]),
            "--end", str(bars["end"]),
            "--timeframe-amount", str(bars["timeframe_amount"]),
            "--feed", str(bars.get("feed", "auto")),
            "--batch-size", str(bars.get("batch_size", 5)),
            "--limit", str(bars.get("limit_symbols", 0)),
        ]
        run(cmd_bars)

    if args.mode == "all_local":
        cmd_ds = [
            py, str(root / "01_build_dataset.py"),
            "--news-path", raw_news,
            "--bars-path", bars_5m,
            "--out-dir", out_dir,
            "--th-5m", str(train["th_5m"]),
            "--th-15m", str(train["th_15m"]),
            "--th-30m", str(train["th_30m"]),
            "--th-60m", str(train["th_60m"]),
        ]
        run(cmd_ds)

        cmd_train = [
            py, str(root / "02_train_event_model.py"),
            "--data-path", dataset_path,
            "--out-dir", model_artifacts,
            "--model-name", str(train["model_name"]),
            "--epochs", str(train["epochs"]),
            "--batch-size", str(train["batch_size"]),
            "--lr", str(train["lr"]),
            "--max-len", str(train["max_len"]),
        ]
        run(cmd_train)

    if args.mode == "kaggle_train":
        print(json.dumps({
            "message": "Esegui questi step su Kaggle dopo aver caricato raw_news.csv e bars_5m.parquet.",
            "dataset_builder": [
                py, str(root / "01_build_dataset.py"),
                "--news-path", cfg["kaggle_paths"]["raw_news"],
                "--bars-path", cfg["kaggle_paths"]["bars_5m"],
                "--out-dir", cfg["project"]["output_dir_kaggle"],
            ],
            "trainer": [
                py, str(root / "02_train_event_model.py"),
                "--data-path", cfg["kaggle_paths"]["dataset_event_model"],
                "--out-dir", cfg["kaggle_paths"]["model_artifacts"],
                "--model-name", str(train["model_name"]),
                "--epochs", str(train["epochs"]),
                "--batch-size", str(train["batch_size"]),
                "--lr", str(train["lr"]),
                "--max-len", str(train["max_len"]),
            ]
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
