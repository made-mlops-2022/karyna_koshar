import json
import os
import pickle
import click
from typing import Dict
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classifier_metrics(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
    }


@click.command("validate")
@click.option("--input-dir", type=click.Path())
@click.option("--model-dir", type=click.Path())
@click.option("--output-dir", type=click.Path())
def validate(input_dir: str, model_dir: str, output_dir: str) -> None:
    model_name = os.listdir(model_dir)[0]
    run_id = model_name[6:-4]
    mlflow.set_tracking_uri("http://0.0.0.0:8000")
    with mlflow.start_run(run_id=run_id):
        X = pd.read_csv(os.path.join(input_dir, "X_val.csv"))
        y = pd.read_csv(os.path.join(input_dir, "y_val.csv"))

        with open(os.path.join(model_dir, f"model_{run_id}.pkl"), "rb") as model_file:
            model = pickle.load(model_file)

        predicts = model.predict(X)
        metrics = classifier_metrics(predicts, y)

        mlflow.log_metric("Accuracy", metrics["accuracy"])
        mlflow.log_metric("F1", metrics["f1"])
        mlflow.log_metric("Precision", metrics["precision"])
        mlflow.log_metric("Recall", metrics["recall"])

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file)


if __name__ == "__main__":
    validate()
