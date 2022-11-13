import json
import click
import logging
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml_project.entities.predict_params import read_predict_params
from ml_project.models.model import read_model


def classifier_metrics(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
    }


def save_predicts(predicts: np.ndarray, output: str) -> str:
    predicts = pd.DataFrame(predicts, columns=["predicts"])
    predicts.to_csv(output, index=False)
    return output


def save_metrics(metrics: Dict[str, float], output: str) -> str:
    with open(output, "w") as metric_file:
        json.dump(metrics, metric_file)
    return output


def read_metrics(input: str) -> Dict[str, float]:
    with open(input, "r") as json_file:
        metrics = json.load(json_file)
    return metrics


def get_stream_handler() -> object:
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))
    return stream_handler


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(get_stream_handler())


@click.command()
@click.argument("config_path")
def predict_model(config_path: str):
    predict_params = read_predict_params(config_path)

    return run_predict_model(predict_params)


def run_predict_model(predict_params):
    logger.info(f"start predict model: {predict_params}")
    model = read_model(predict_params.model_path)
    df_val = pd.read_csv(predict_params.features_val_path)
    predicts = model.predict(df_val)

    logger.info("saving predicts")
    path_to_predicts = save_predicts(predicts, predict_params.predicts_path)

    return path_to_predicts


if __name__ == "__main__":
    predict_model()
