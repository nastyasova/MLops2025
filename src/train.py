import argparse
import logging
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_and_preprocess_data
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def load_data(config):
    """Загружает и токенизирует данные на основе конфигурации."""
    logging.info("Загружаем и подготавливаем данные...")
    train_ds, test_ds, tokenizer = load_and_preprocess_data(config)

    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")

    if max_train:
        train_ds = train_ds.select(range(min(len(train_ds), int(max_train))))
    if max_eval:
        test_ds = test_ds.select(range(min(len(test_ds), int(max_eval))))

    logging.info(f"Using subsets: train={len(train_ds)}, eval={len(test_ds)}")
    return train_ds, test_ds, tokenizer


def prepare_model(config):
    """Создаёт и настраивает модель по конфигу."""
    model_name = config["model_name"]
    logging.info(f"Загружаем модель: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if config.get("freeze_base", False):
        logging.info("freeze_base=true: freezing base weights (training only classifier head)")
        for name, p in model.named_parameters():
            if not (name.startswith("classifier") or name.startswith("pre_classifier")):
                p.requires_grad = False

    return model


def train_model(model, train_ds, test_ds, config):
    """Настраивает Trainer и запускает обучение модели."""
    lr = float(config["lr"])
    batch_size = int(config["batch_size"])
    num_epochs = float(config["num_epochs"])
    output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",  
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir="logs",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds, 
        compute_metrics=compute_metrics,
    )

    logging.info("Начинаем обучение...")
    trainer.train()
    return trainer


def save_model(model, tokenizer, output_dir):
    """Сохраняет обученную модель и токенизатор (HF-compatible)."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Модель и токенизатор сохранены в: {output_dir}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_ds, test_ds, tokenizer = load_data(config)
    model = prepare_model(config)
    trainer = train_model(model, train_ds, test_ds, config)
    save_model(trainer.model, tokenizer, config["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
