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


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    eval_metrics_path = cfg.get("eval_metrics_path", os.path.join(output_dir, "eval_metrics.yaml"))

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
    os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)

    with open(eval_metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)

    logging.info(f"Saved eval metrics to: {eval_metrics_path}")
    logging.info("Evaluate stage done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    args = ap.parse_args()
    main(args.config)
