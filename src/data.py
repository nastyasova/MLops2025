import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(config):
    logging.info(f"Загружаем датасет: {config['dataset_name']}")
    dataset = load_dataset(config["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    logging.info("Выполняем токенизацию...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    logging.info(
        f"Данные загружены: train={len(train_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, test_dataset, tokenizer

