import os
import logging
import subprocess
import yaml
import torch

import mlflow
import mlflow.transformers

from transformers import (
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
    return {"accuracy": float(acc), "f1": float(f1)}


def _run_cmd(cmd: list[str]) -> str:
    """Run shell command and return stdout (stripped). Never raises; returns '' on failure."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def get_git_commit() -> str:
    return _run_cmd(["git", "rev-parse", "HEAD"])


def get_dvc_outputs_hashes() -> dict:
    """
    Best-effort: collect DVC hashes for outputs (links MLflow run <-> DVC state).
    Works if `dvc` is installed and repo is initialized.
    """
    hashes = {}
    if os.path.exists("dvc.lock"):
        try:
            with open("dvc.lock", "r", encoding="utf-8") as f:
                lock = yaml.safe_load(f) or {}
            stages = lock.get("stages", {})
            for stage_name, stage in stages.items():
                outs = stage.get("outs", []) or []
                for out in outs:
                    path = out.get("path")
                    md5 = out.get("md5")
                    if path and md5:
                        hashes[f"{stage_name}:{path}"] = md5
        except Exception:
            pass
    return hashes


def load_data(config):
    logging.info("Загружаем и подготавливаем данные...")
    train_ds, test_ds, tokenizer = load_and_preprocess_data(config)

    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")

    if max_train:
        train_ds = train_ds.select(range(min(len(train_ds), int(max_train))))
    if max_eval:
        test_ds = test_ds.select(range(min(len(test_ds), int(max_eval))))

    logging.info(f"Using subsets: train={len(train_ds)}, eval={len(test_ds)}")
    return train_ds, test_ds, tokenizer


def prepare_model(config):
    model_name = config["model_name"]
    logging.info(f"Загружаем модель: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if config.get("freeze_base", False):
        logging.info("freeze_base=true: freezing base weights (training only classifier head)")
        for name, p in model.named_parameters():
            if not (name.startswith("classifier") or name.startswith("pre_classifier")):
                p.requires_grad = False

    return model


def train_and_eval(model, tokenizer, train_ds, test_ds, config):
    lr = float(config["lr"])
    batch_size = int(config["batch_size"])
    num_epochs = int(config["num_epochs"])
    output_dir = config["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir="logs",
        logging_steps=50,
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

    logging.info("Оцениваем модель на тестовых данных...")
    metrics = trainer.evaluate()

    clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return trainer, clean_metrics


def save_reports(metrics: dict, metrics_path: str, eval_metrics_path: str):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, allow_unicode=True)
    with open(eval_metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, allow_unicode=True)
    logging.info(f"Saved metrics to: {metrics_path} and {eval_metrics_path}")


def main(config_path="configs/train_config.yaml"):
    config = load_config(config_path)


    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("imdb-sentiment")

    mlflow.transformers.autolog(log_models=False)  # модели логируем сами как artifacts


    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    train_ds, test_ds, tokenizer = load_data(config)
    model = prepare_model(config)

    run_name = f"{config.get('model_name','model')}_bs{config.get('batch_size')}_lr{config.get('lr')}_ep{config.get('num_epochs')}"

    with mlflow.start_run(run_name=run_name):

        for k, v in config.items():
            if isinstance(v, (dict, list)):
                mlflow.log_param(k, str(v))
            else:
                mlflow.log_param(k, v)


        git_commit = get_git_commit()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)

        dvc_hashes = get_dvc_outputs_hashes()
        if dvc_hashes:

            for hk, hv in dvc_hashes.items():
                safe_key = f"dvc/{hk}".replace(":", "/")
                mlflow.set_tag(safe_key, hv)



        trainer, metrics = train_and_eval(model, tokenizer, train_ds, test_ds, config)

        out_dir = config["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logging.info(f"Модель и токенизатор сохранены в: {out_dir}")


        metrics_path = os.path.join("reports", "metrics.yaml")
        eval_metrics_path = os.path.join("reports", "eval_metrics.yaml")
        save_reports(metrics, metrics_path, eval_metrics_path)

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

        if os.path.exists("dvc.lock"):
            mlflow.log_artifact("dvc.lock")
        if os.path.exists("dvc.yaml"):
            mlflow.log_artifact("dvc.yaml")

        if os.path.exists(metrics_path):
            mlflow.log_artifact(metrics_path)
        if os.path.exists(eval_metrics_path):
            mlflow.log_artifact(eval_metrics_path)

        if os.path.isdir(out_dir):
            mlflow.log_artifacts(out_dir, artifact_path="model_dir")

    return metrics


if __name__ == "__main__":
    main()
