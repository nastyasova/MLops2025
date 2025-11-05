import pytest
import torch
from src.data import load_and_preprocess_data
from src.utils import load_config

@pytest.fixture(scope="module")
def data_setup():
    """Загружаем данные и токенизатор один раз для всех тестов."""
    config = load_config("configs/train_config.yaml")
    train_ds, test_ds, tokenizer = load_and_preprocess_data(config)
    return train_ds, test_ds, tokenizer


def test_dataset_not_empty(data_setup):
    train_ds, test_ds, _ = data_setup
    assert len(train_ds) > 0, "Обучающая выборка пуста!"
    assert len(test_ds) > 0, "Тестовая выборка пуста!"


def test_dataset_fields_exist(data_setup):
    train_ds, _, _ = data_setup
    expected_fields = {"input_ids", "attention_mask", "label"}
    assert expected_fields.issubset(train_ds.features.keys()), "В датасете не хватает нужных полей!"


def test_label_values_in_range(data_setup):
    train_ds, _, _ = data_setup
    labels = [x["labels"] for x in train_ds[:100]]
    assert all(l in [0, 1] for l in labels), "Метки выходят за допустимый диапазон (0 или 1)!"


def test_tokenizer_output_type(data_setup):
    _, _, tokenizer = data_setup
    tokens = tokenizer("Hello world!")["input_ids"]
    assert isinstance(tokens, list), "Токенайзер должен возвращать список!"
    assert all(isinstance(t, int) for t in tokens), "Все токены должны быть целыми числами!"


def test_input_id_range(data_setup):
    _, _, tokenizer = data_setup
    vocab_size = tokenizer.vocab_size
    sample_ids = tokenizer("test sentence")["input_ids"]
    assert all(0 <= t < vocab_size for t in sample_ids), "Некоторые токены выходят за пределы словаря!"
