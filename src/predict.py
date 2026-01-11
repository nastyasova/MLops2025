from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)

    # путь до сохранённой модели (по умолчанию как у тебя в проекте)
    p.add_argument("--model_dir", type=str, default="models/distilbert-imdb")

    # ожидаемая колонка с текстом во входном CSV
    p.add_argument("--text_col", type=str, default="text")

    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise ValueError(
            f"Expected column '{args.text_col}' in {input_path}, got {list(df.columns)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    texts = df[args.text_col].astype(str).tolist()

    preds = []
    probs = []

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        out = model(**enc)
        logits = out.logits
        p = torch.softmax(logits, dim=-1)

        pred = torch.argmax(p, dim=-1).cpu().numpy()
        pos_prob = p[:, 1].cpu().numpy() if p.shape[1] >= 2 else p[:, 0].cpu().numpy()

        preds.extend(pred.tolist())
        probs.extend(pos_prob.tolist())

    out_df = df.copy()
    out_df["pred_label"] = preds
    out_df["pred_proba_pos"] = probs
    out_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
