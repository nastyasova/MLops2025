import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.get_experiment_by_name("imdb-sentiment")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

runs.to_csv("reports/mlflow_runs.csv", index=False)
print("Saved reports/mlflow_runs.csv")
