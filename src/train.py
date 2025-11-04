from src.data import load_and_preprocess_data
import yaml

with open("configs/train_config.yaml") as f:
    config = yaml.safe_load(f)

train_ds, test_ds, tok = load_and_preprocess_data(config)
print(len(train_ds), len(test_ds))
print(tok("I love this movie!"))

