import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from ml_project.entities.feature_params import FeatureParams
from ml_project.entities.preprocessing_params import PreprocessingParams
from ml_project.features.custom_transformer import CategoricalTransformer


def build_categorical_pipeline(params: PreprocessingParams) -> Pipeline:
    pipeline_steps = []

    if params.use_custom_transformer:
        pipeline_steps.append(("transformer", CategoricalTransformer()))
    else:
        pipeline_steps.append(("ohe", OneHotEncoder(drop="first")))

    categorical_pipeline = Pipeline(steps=pipeline_steps)
    return categorical_pipeline


def build_numerical_pipeline(params: PreprocessingParams) -> Pipeline:
    pipeline_steps = []

    if params.use_scaler:
        if params.scaler == "StandardScaler":
            pipeline_steps.append(("scaler", StandardScaler()))
        elif params.scaler == "MinMaxScaler":
            pipeline_steps.append(("scaler", MinMaxScaler()))
        else:
            raise NotImplementedError()

    num_pipeline = Pipeline(steps=pipeline_steps)
    return num_pipeline


def process_data(transformer: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(data)


def build_transformer(
    feature_params: FeatureParams, preprocessing_params: PreprocessingParams
) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(preprocessing_params),
                feature_params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(preprocessing_params),
                feature_params.numerical_features,
            ),
        ]
    )
    return transformer
