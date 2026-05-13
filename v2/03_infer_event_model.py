import os
import json
import html
import re
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def normalize_text(x: str) -> str:
    x = html.unescape(str(x or ""))
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"https?://\\S+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


class MultiTaskEventModel(nn.Module):
    def __init__(self, model_name, n_numeric, n_event_classes, n_horizons=4, n_move_classes=3, dropout=0.25):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.num_net = nn.Sequential(
            nn.Linear(n_numeric, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 32, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(128, n_event_classes)
        self.move_heads = nn.ModuleList([nn.Linear(128, n_move_classes) for _ in range(n_horizons)])
        self.severity_head = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, features):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        num = self.num_net(features)
        z = self.fusion(torch.cat([cls, num], dim=1))
        event_logits = self.event_head(z)
        move_logits = torch.stack([head(z) for head in self.move_heads], dim=1)
        severity = self.severity_head(z).squeeze(-1)
        return event_logits, move_logits, severity


def main(args):
    with open(os.path.join(args.model_dir, "feature_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    mean = np.load(os.path.join(args.model_dir, "scaler_mean.npy"))
    scale = np.load(os.path.join(args.model_dir, "scaler_scale.npy"))

    model = MultiTaskEventModel(
        cfg["model_name"],
        n_numeric=len(cfg["numeric_cols"]),
        n_event_classes=len(cfg["event_classes"]),
        n_horizons=len(cfg["horizon_labels"]),
    )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location="cpu"))
    model.eval()

    text = normalize_text(args.title + " [SEP] " + args.summary)
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=cfg["max_len"], return_tensors="pt")

    values = np.array([
        args.hour_utc, args.weekday, args.month, len(text),
        args.event_price, args.event_volume, args.vol_z, args.recent_abs_ret_mean
    ], dtype=float)
    values = (values - mean) / np.where(scale == 0, 1.0, scale)
    feats = torch.tensor(values.reshape(1, -1), dtype=torch.float32)

    with torch.no_grad():
        ev, mv, sev = model(enc["input_ids"], enc["attention_mask"], feats)
        ev_p = torch.softmax(ev, dim=1)[0].numpy()
        mv_p = torch.softmax(mv, dim=2)[0].numpy()

    out = {
        "event_probs": {cls: float(p) for cls, p in zip(cfg["event_classes"], ev_p)},
        "move_probs": {
            h: {"short": float(p[0]), "flat": float(p[1]), "long": float(p[2])}
            for h, p in zip(cfg["horizon_labels"], mv_p)
        },
        "severity": float(sev.item()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--hour-utc", type=float, default=14)
    ap.add_argument("--weekday", type=float, default=2)
    ap.add_argument("--month", type=float, default=5)
    ap.add_argument("--event-price", type=float, default=100)
    ap.add_argument("--event-volume", type=float, default=1000000)
    ap.add_argument("--vol-z", type=float, default=0.0)
    ap.add_argument("--recent-abs-ret-mean", type=float, default=0.01)
    main(ap.parse_args())
