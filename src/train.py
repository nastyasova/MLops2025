import os
import logging
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)
    model_name = config.get("model_name", "distilbert-base-uncased")
    num_epochs = config.get("num_train_epochs", 1)
    batch_size = config.get("batch_size", 8)
    lr = config.get("learning_rate", 2e-5)
    seed = config.get("random_seed", 42)
    output_dir = config.get("output_dir", "models/distilbert-imdb")
    torch.manual_seed(seed)
    logging.info("Загружаем датасет...")
    dataset = load_dataset("stanfordnlp/imdb")
    logging.info("Загружаем токенизатор и модель...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    logging.info("Создаём аргументы обучения...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=100,
        report_to="none",
    )

    logging.info("Инициализируем Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    logging.info("Начинаем обучение...")
    trainer.train()

    logging.info("Проверяем модель...")
    metrics = trainer.evaluate()
    logging.info(f"Метрики на тесте: {metrics}")

    logging.info("Сохраняем модель...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logging.info(f"Модель успешно сохранена в: {output_dir}")

if __name__ == "__main__":
    main()
