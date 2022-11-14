import click
import mlflow
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities.train_pipeline_params import read_train_pipeline_params
from ml_project.features.features import build_transformer, process_data
from ml_project.models.model import (
    train_model,
    model_pipeline,
    save_model,
    read_model,
)
from ml_project.predict import classifier_metrics, save_metrics, read_metrics
from ml_project.entities.train_pipeline_params import TrainPipelineParams


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
def train_pipeline(config_path: str) -> str:
    train_pipeline_params = read_train_pipeline_params(config_path)

    if train_pipeline_params.use_mlflow:
        with mlflow.start_run():
            path_to_model, path_to_metrics = run_train_pipeline(train_pipeline_params)
            model = read_model(path_to_model)
            metrics = read_metrics(path_to_metrics)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("Accuracy", metrics["accuracy"])
            mlflow.log_metric("F1", metrics["f1"])
            mlflow.log_metric("Precision", metrics["precision"])
            mlflow.log_metric("Recall", metrics["recall"])
            mlflow.log_artifact(config_path)
    else:
        return run_train_pipeline(train_pipeline_params)


def run_train_pipeline(train_pipeline_params: TrainPipelineParams) -> str:
    logger.info(f"start train pipeline: {train_pipeline_params}")
    data = pd.read_csv(train_pipeline_params.input_data_path)
    logger.info(f"data shape: {data.shape}")
    df_train, df_val = train_test_split(
        data,
        test_size=train_pipeline_params.splitting_params.val_size,
        random_state=train_pipeline_params.splitting_params.random_state,
    )

    logger.info(f"df_train shape: {df_train.shape}")
    logger.info(f"df_val shape: {df_val.shape}")
    target_train = df_train[train_pipeline_params.feature_params.target_col]
    target_val = df_val[train_pipeline_params.feature_params.target_col]
    df_train = df_train.drop(train_pipeline_params.feature_params.target_col, axis=1)
    df_val = df_val.drop(train_pipeline_params.feature_params.target_col, axis=1)

    logger.info("build transformer")
    transformer = build_transformer(
        train_pipeline_params.feature_params, train_pipeline_params.preprocessing_params
    )
    transformer.fit(df_train)

    features_train = process_data(transformer, df_train)
    logger.info(f"features_train shape: {features_train.shape}")

    logger.info("train model")
    model = train_model(
        features_train, target_train, train_pipeline_params.train_params
    )

    logger.info("saving model and features_val")
    pipeline_result = model_pipeline(model, transformer)
    path_to_model = save_model(pipeline_result, train_pipeline_params.output_model_path)
    df_val.to_csv(train_pipeline_params.features_val_path, index=False)

    logger.info("start predict model")
    predicts = pipeline_result.predict(df_val)
    metrics_result = classifier_metrics(predicts, target_val)
    logger.info(f"metrics result: {metrics_result}")

    logger.info("saving metrics")
    path_to_metrics = save_metrics(metrics_result, train_pipeline_params.metric_path)

    return path_to_model, path_to_metrics


if __name__ == "__main__":
    train_pipeline()
