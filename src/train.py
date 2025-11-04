from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.data import load_and_preprocess_data
from configs.config import load_config

config = load_config()
train_ds, test_ds, tokenizer = load_and_preprocess_data(config)

model = AutoModelForSequenceClassification.from_pretrained(
    config["model"]["name"], num_labels=2
)

training_args = TrainingArguments(
    output_dir="./models/checkpoints",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

trainer.train()
