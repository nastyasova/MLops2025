from src.data import load_and_preprocess_data
import yaml


with open("configs/train_config.yaml") as f:
    config = yaml.safe_load(f)
train_ds, test_ds, tokenizer = load_and_preprocess_data(config)
print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
example = tokenizer("I love this movie!")["input_ids"][:10]
print("Пример токенов:", example)
