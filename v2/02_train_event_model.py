import os
import json
import math
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


NUMERIC_COLS = [
    "hour_utc", "weekday", "month", "text_len",
    "event_price", "event_volume", "vol_z", "recent_abs_ret_mean",
]
EVENT_LABEL_COL = "event_type_weak"
HORIZON_LABELS = ["y_cls_5m", "y_cls_15m", "y_cls_30m", "y_cls_60m"]


class NewsEventDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, scaler, event_encoder):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.scaler = scaler
        self.event_encoder = event_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        x_num = row[NUMERIC_COLS].astype(float).values.reshape(1, -1)
        x_num = self.scaler.transform(x_num)[0]
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": torch.tensor(x_num, dtype=torch.float32),
            "y_event": torch.tensor(self.event_encoder.transform([row[EVENT_LABEL_COL]])[0], dtype=torch.long),
        }
        for c in HORIZON_LABELS:
            item[c] = torch.tensor(int(row[c]), dtype=torch.long)
        return item


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


def compute_metrics_event(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def compute_metrics_move(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def evaluate(model, loader, device):
    model.eval()
    ev_true, ev_pred = [], []
    hz_true = [[] for _ in HORIZON_LABELS]
    hz_pred = [[] for _ in HORIZON_LABELS]
    sev_true, sev_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            y_event = batch["y_event"].to(device)
            y_h = [batch[c].to(device) for c in HORIZON_LABELS]
            y_sev = batch["severity_score_weak"].to(device) if "severity_score_weak" in batch else None

            event_logits, move_logits, sev = model(input_ids, attention_mask, features)
            ev_true.extend(y_event.cpu().numpy().tolist())
            ev_pred.extend(event_logits.argmax(1).cpu().numpy().tolist())
            for i in range(len(HORIZON_LABELS)):
                hz_true[i].extend(y_h[i].cpu().numpy().tolist())
                hz_pred[i].extend(move_logits[:, i, :].argmax(1).cpu().numpy().tolist())
            if y_sev is not None:
                sev_true.extend(y_sev.cpu().numpy().tolist())
                sev_pred.extend(sev.cpu().numpy().tolist())

    out = {"event": compute_metrics_event(ev_true, ev_pred)}
    for i, c in enumerate(HORIZON_LABELS):
        out[c] = compute_metrics_move(hz_true[i], hz_pred[i])
    return out


def collate_with_severity(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([x[k] for x in batch])
    return out


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.data_path)
    df = df.dropna(subset=["text", EVENT_LABEL_COL] + NUMERIC_COLS + HORIZON_LABELS).copy()
    df["severity_score_weak"] = df.get("severity_score_weak", 0.0).astype(float)

    train_df, valid_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df[EVENT_LABEL_COL],
    )

    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLS].astype(float).values)

    event_encoder = LabelEncoder()
    event_encoder.fit(df[EVENT_LABEL_COL].astype(str).values)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = NewsEventDataset(train_df, tokenizer, args.max_len, scaler, event_encoder)
    valid_ds = NewsEventDataset(valid_df, tokenizer, args.max_len, scaler, event_encoder)

    def add_severity(ds):
        orig_get = ds.__getitem__
        def _wrap(i):
            item = orig_get(i)
            item["severity_score_weak"] = torch.tensor(float(ds.df.iloc[i]["severity_score_weak"]), dtype=torch.float32)
            return item
        ds.__getitem__ = _wrap
        return ds

    train_ds = add_severity(train_ds)
    valid_ds = add_severity(valid_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_with_severity)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_with_severity)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskEventModel(
        model_name=args.model_name,
        n_numeric=len(NUMERIC_COLS),
        n_event_classes=len(event_encoder.classes_),
        n_horizons=len(HORIZON_LABELS),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    ce_event = nn.CrossEntropyLoss()
    ce_move = nn.CrossEntropyLoss()
    mse_sev = nn.MSELoss()

    best_score = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            y_event = batch["y_event"].to(device)
            y_moves = [batch[c].to(device) for c in HORIZON_LABELS]
            y_sev = batch["severity_score_weak"].to(device)

            event_logits, move_logits, sev = model(input_ids, attention_mask, features)
            loss = ce_event(event_logits, y_event)
            for i in range(len(HORIZON_LABELS)):
                loss = loss + ce_move(move_logits[:, i, :], y_moves[i])
            loss = loss + 0.5 * mse_sev(sev, y_sev)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        metrics = evaluate(model, valid_loader, device)
        score = metrics["event"]["f1_macro"] + metrics["y_cls_30m"]["f1_macro"]
        print(json.dumps({"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), "valid": metrics}, ensure_ascii=False))

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

    tokenizer.save_pretrained(os.path.join(args.out_dir, "tokenizer"))
    with open(os.path.join(args.out_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "numeric_cols": NUMERIC_COLS,
            "event_label_col": EVENT_LABEL_COL,
            "horizon_labels": HORIZON_LABELS,
            "model_name": args.model_name,
            "max_len": args.max_len,
            "event_classes": event_encoder.classes_.tolist(),
        }, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(args.out_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(args.out_dir, "scaler_scale.npy"), scaler.scale_)
    print(json.dumps({"best_model": best_path, "best_score": best_score}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="ProsusAI/finbert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=192)
    main(ap.parse_args())
