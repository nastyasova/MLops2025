import os
import pytest
import torch
from transformers import Trainer
from src.train import compute_metrics
from src.utils import load_config
from src.data import load_and_preprocess_data
from transformers import AutoModelForSequenceClassification, TrainingArguments


@pytest.fixture(scope="module")
def small_training_setup():
    """Мини-настройка для быстрой проверки пайплайна."""
    config = load_config("configs/train_config.yaml")
    train_ds, test_ds, tokenizer = load_and_preprocess_data(config)

    # Берём только первые 200 строк, чтобы ускорить тест
    small_train = train_ds.select(range(200))
    small_test = test_ds.select(range(100))

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2
    )

    args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=0.1,  # один мини-проход
        learning_rate=float(config["lr"]),
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train,
        eval_dataset=small_test,
        compute_metrics=compute_metrics,
    )

    return trainer, config


def test_trainer_initialization(small_training_setup):
    trainer, config = small_training_setup
    assert isinstance(trainer, Trainer), "Trainer не был создан!"
    assert trainer.model is not None, "Модель не инициализирована!"
    assert os.path.isdir(config["output_dir"]) or not os.path.exists(config["output_dir"]), \
        "Некорректная директория для сохранения модели!"


def test_training_and_saving(small_training_setup):
    trainer, config = small_training_setup
    trainer.train()
    trainer.save_model(config["output_dir"])

    expected_files = [
    "config.json",
    "trainer_state.json",
    ]
    weight_files = ["pytorch_model.bin", "model.safetensors"]
    
    found = False
    for wf in weight_files:
        if os.path.exists(os.path.join(config["output_dir"], wf)):
            found = True
            break
    assert found, f"Файл весов не найден ни в одном из форматов: {weight_files}"
    
    for file in expected_files:
        path = os.path.join(config["output_dir"], file)
        assert os.path.exists(path), f"Файл не найден: {path}"


def test_evaluation_metrics(small_training_setup):
    trainer, _ = small_training_setup
    metrics = trainer.evaluate()
    assert isinstance(metrics, dict), "Метрики должны возвращаться в виде словаря!"
    assert "eval_loss" in metrics or "eval_accuracy" in metrics, "Отсутствуют ключевые метрики!"
    print("Метрики:", metrics)
