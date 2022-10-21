import json
import click
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from enities.predict_params import read_predict_params
from models.model import read_model


def classifier_metrics(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts)
    }


def save_predicts(predicts: np.ndarray, output: str) -> str:
    predicts = pd.DataFrame(predicts, columns=['predicts'])
    return predicts.to_csv(output, index=False)


def save_metrics(metrics: Dict[str, float], output: str) -> str:
    with open(output, "w") as metric_file:
        json.dump(metrics, metric_file)
    return output


@click.command()
@click.argument("config_path")
def predict_model(config_path: str):
    predict_params = read_predict_params(config_path)

    model = read_model(predict_params.model_path)
    target_val = pd.read_csv(predict_params.target_val_path)
    df_val = pd.read_csv(predict_params.features_val_path)

    predicts = model.predict(df_val)

    metrics_result  = classifier_metrics(
        predicts,
        target_val
    )

    path_to_predicts = save_predicts(predicts, predict_params.predicts_path)
    path_to_metrics = save_metrics(metrics_result, predict_params.metric_path)

    return path_to_predicts, path_to_metrics


if __name__ == "__main__":
    predict_model()
