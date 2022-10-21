import click
import pandas as pd
from sklearn.model_selection import train_test_split

from enities.train_pipeline_params import read_train_pipeline_params
from features.features import build_transformer, process_data
from models.model import (
    train_model, 
    model_pipeline, 
    save_model
)


@click.command()
@click.argument("config_path")
def train_pipeline(config_path: str):
    train_pipeline_params = read_train_pipeline_params(config_path)

    return run_train_pipeline(train_pipeline_params)


def run_train_pipeline(train_pipeline_params):

    data = pd.read_csv(train_pipeline_params.input_data_path)

    df_train, df_val = train_test_split(
        data, 
        test_size=train_pipeline_params.splitting_params.val_size,
        random_state=train_pipeline_params.splitting_params.random_state
    )

    target_train = df_train[train_pipeline_params.feature_params.target_col]
    target_val = df_val[train_pipeline_params.feature_params.target_col]    
    df_train = df_train.drop(train_pipeline_params.feature_params.target_col, axis=1)
    df_val = df_val.drop(train_pipeline_params.feature_params.target_col, axis=1)

    transformer = build_transformer(
        train_pipeline_params.feature_params,
        train_pipeline_params.preprocessing_params
        )
    
    transformer.fit(df_train)
    features_train = process_data(transformer, df_train)

    model = train_model(
        features_train, 
        target_train, 
        train_pipeline_params.train_params
    )

    pipeline_result = model_pipeline(model, transformer)
    path_to_model = save_model(pipeline_result, train_pipeline_params.output_model_path)
    path_to_features_val = df_val.to_csv(train_pipeline_params.features_val_path, index=False) 
    path_to_target_val = target_val.to_csv(train_pipeline_params.target_val_path, index=False)

    return path_to_model, path_to_features_val, path_to_target_val


if __name__ == "__main__":
    train_pipeline()