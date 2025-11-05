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
from src.data import load_and_preprocess_data
from src.utils import load_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    logging.info("Загружаем конфигурацию...")
    config = load_config("configs/train_config.yaml")

    model_name = config["model_name"]
    output_dir = config["output_dir"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = float(config["lr"])
    seed = config["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Загружаем и подготавливаем данные...")
    train_ds, test_ds, tokenizer = load_and_preprocess_data(config)
    logging.info(f"Размер обучающей выборки: {len(train_ds)}, тестовой: {len(test_ds)}")
    logging.info(f"Загружаем модель: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir="logs",
        logging_steps=100,
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
    logging.info("Проверяем модель на тестовых данных...")
    metrics = trainer.evaluate()
    logging.info(f"Метрики: {metrics}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Модель и токенизатор сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
