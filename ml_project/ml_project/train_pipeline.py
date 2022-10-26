import click
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities.train_pipeline_params import read_train_pipeline_params
from ml_project.features.features import build_transformer, process_data
from ml_project.models.model import train_model, model_pipeline, save_model


def get_stream_handler():
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
def train_pipeline(config_path: str):

    train_pipeline_params = read_train_pipeline_params(config_path)

    return run_train_pipeline(train_pipeline_params)


def run_train_pipeline(train_pipeline_params):
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

    logger.info("saving model, features_val and target_val")
    pipeline_result = model_pipeline(model, transformer)
    path_to_model = save_model(pipeline_result, train_pipeline_params.output_model_path)
    df_val.to_csv(train_pipeline_params.features_val_path, index=False)
    target_val.to_csv(train_pipeline_params.target_val_path, index=False)

    return path_to_model


if __name__ == "__main__":
    train_pipeline()
