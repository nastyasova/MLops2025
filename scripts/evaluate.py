import os
import yaml
import argparse
import logging
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.data import load_and_preprocess_data
from src.train import compute_metrics


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def main(config_path: str, metrics_path: str, eval_metrics_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]

    train_ds, test_ds, _ = load_and_preprocess_data(cfg)

    logging.info(f"Loading model from: {output_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_eval_tmp"),
        per_device_eval_batch_size=int(cfg.get("batch_size", 8)),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)

    # можно один и тот же словарь писать в оба файла,
    # или разделить "train metrics" и "eval metrics" — для простоты одинаково:
    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)

    with open(eval_metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)

    logging.info(f"Saved metrics to: {metrics_path} and {eval_metrics_path}")
    logging.info("Evaluate stage done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--metrics_path", required=True)
    ap.add_argument("--eval_metrics_path", required=True)
    args = ap.parse_args()
    main(args.config, args.metrics_path, args.eval_metrics_path)
