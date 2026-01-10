import os
import logging
import torch
import yaml
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


def load_data(config):
    """Загружает и токенизирует данные на основе конфигурации."""
    logging.info("Загружаем и подготавливаем данные...")
    train_ds, test_ds, tokenizer = load_and_preprocess_data(config)
    logging.info(f"Размер обучающей выборки: {len(train_ds)}, тестовой: {len(test_ds)}")
    return train_ds, test_ds, tokenizer


def prepare_model(config):
    """Создаёт и настраивает модель по конфигу."""
    model_name = config["model_name"]
    logging.info(f"Загружаем модель: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model


def train_model(model, train_ds, test_ds, config):
    """Настраивает Trainer и запускает обучение модели."""
    lr = float(config["lr"])
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    output_dir = config["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
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
    return trainer


def evaluate_model(trainer):
    """Проверяет обученную модель и возвращает метрики."""
    logging.info("Проверяем модель на тестовых данных...")
    metrics = trainer.evaluate()
    logging.info(f"Метрики: {metrics}")
    return metrics



def save_model(model, tokenizer, output_dir):
    """Сохраняет обученную модель и токенизатор."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Модель и токенизатор сохранены в: {output_dir}")



def main(config_path="configs/train_config.yaml"):
    config = load_config(config_path)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds, test_ds, tokenizer = load_data(config)
    model = prepare_model(config)
    trainer = train_model(model, train_ds, test_ds, config)
    metrics = evaluate_model(trainer)
    metrics_path = os.path.join(config["output_dir"], "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)
    logging.info(f"Метрики сохранены в {metrics_path}")
    save_model(model, tokenizer, config["output_dir"])
    return metrics


if __name__ == "__main__":
    main()
