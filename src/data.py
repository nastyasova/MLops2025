import logging
from datasets import load_dataset
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


def load_and_preprocess_data(config):
    dataset_name = config.get("dataset_name", "stanfordnlp/imdb")
    model_name = config.get("model_name", "distilbert-base-uncased")
    logging.info(f"Загружаем датасет: {dataset_name}")
    dataset = load_dataset(dataset_name)
    logging.info(f"Загружаем токенизатор: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    logging.info("Токенизация данных...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    logging.info(f"Обучающая выборка: {len(train_dataset)}, тестовая: {len(test_dataset)}")

    return train_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    import yaml
    with open("configs/train_config.yaml") as f:
        cfg = yaml.safe_load(f)
    load_and_preprocess_data(cfg)
