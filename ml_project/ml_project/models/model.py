import pickle
from typing import Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_project.entities.train_params import TrainParams

SklearnClassifier = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainParams
) -> SklearnClassifier:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()

    model.fit(features, target)

    return model


def model_pipeline(
    model: SklearnClassifier, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("preprocessing_part", transformer), ("model_part", model)])


def save_model(model: object, output: str) -> str:
    with open(output, "wb") as model_file:
        pickle.dump(model, model_file)
    return output


def read_model(input: str) -> object:
    with open(input, "rb") as model_file:
        model = pickle.load(model_file)
    return model
