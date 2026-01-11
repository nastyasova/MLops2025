import os
import yaml
import argparse
import logging
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    model_name = cfg["model_name"]
    raw_dir = cfg["raw_data_dir"]
    processed_dir = cfg["processed_data_dir"]
    max_length = int(cfg.get("max_length", 256))

    os.makedirs(os.path.dirname(raw_dir), exist_ok=True)
    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)

    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        logging.info(f"Downloading dataset {dataset_name} and saving RAW to: {raw_dir}")
        ds = load_dataset(dataset_name)
        ds.save_to_disk(raw_dir)
    else:
        logging.info(f"RAW already exists, loading from disk: {raw_dir}")
        ds = load_from_disk(raw_dir)

    logging.info(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tok(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    logging.info(f"Tokenizing and saving PROCESSED to: {processed_dir}")
    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.save_to_disk(processed_dir)

    logging.info("Prepare stage done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    args = ap.parse_args()
    main(args.config)
