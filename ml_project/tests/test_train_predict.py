import os
import json
import pandas as pd
import pytest
from click.testing import CliRunner

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from tests.test_helper import (
    generate_fake_data,
    numerical_features,
    categorical_features,
    target_col,
    create_config_train,
    create_config_predict,
)

from ml_project.train_pipeline import run_train_pipeline
from ml_project.predict import run_predict_model
from ml_project.models.model import (
    train_model,
    model_pipeline,
    read_model,
    save_model,
)
from ml_project.features.features import CategoricalTransformer
from ml_project.features.features import (
    build_categorical_pipeline,
    build_numerical_pipeline,
    process_data,
    build_transformer,
)
from ml_project.predict import (
    classifier_metrics,
    save_predicts,
    save_metrics,
)

from ml_project.entities.train_params import TrainParams
from ml_project.entities.feature_params import FeatureParams
from ml_project.entities.preprocessing_params import PreprocessingParams


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_valid_all_project(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        fake_data = generate_fake_data(200)
        os.mkdir("data")
        fake_data.to_csv("data/heart_cleveland_upload.csv", index=False)

        os.mkdir("models")
        path_to_model = run_train_pipeline(create_config_train())
        assert os.path.exists(path_to_model)

        path_to_predicts, path_to_metrics = run_predict_model(create_config_predict())
        assert os.path.exists(path_to_predicts)
        assert os.path.exists(path_to_metrics)

        with open("models/metrics.json", "r") as json_file:
            metrics = json.load(json_file)

        assert 1 >= metrics["accuracy"] >= 0
        assert 1 >= metrics["f1"] >= 0
        assert 1 >= metrics["precision"] >= 0
        assert 1 >= metrics["recall"] >= 0

        predicts = pd.read_csv("models/predicts.csv")
        assert predicts.shape == (40, 1)


def test_model(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        fake_data = generate_fake_data(200)
        target_train = fake_data["condition"]
        df_train = fake_data.drop("condition", axis=1)

        model = train_model(
            df_train,
            target_train,
            TrainParams(model_type="RandomForestClassifier", random_state=42),
        )
        assert isinstance(model, RandomForestClassifier)

        transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical_pipeline",
                    Pipeline(steps=[("ohe", OneHotEncoder(drop="first"))]),
                    categorical_features(),
                ),
                (
                    "numerical_pipeline",
                    Pipeline(steps=[("scaler", StandardScaler())]),
                    numerical_features(),
                ),
            ]
        )
        pipeline = model_pipeline(model, transformer)

        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline["model_part"], RandomForestClassifier)
        assert isinstance(pipeline["preprocessing_part"], ColumnTransformer)
        assert pipeline["preprocessing_part"] == transformer

        os.mkdir("models")
        path_to_model = save_model(pipeline, "models/model.pkl")
        assert os.path.exists(path_to_model)

        model = read_model("models/model.pkl")
        assert isinstance(model["model_part"], RandomForestClassifier)
        assert isinstance(model["preprocessing_part"], ColumnTransformer)


def test_features(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        fake_data = generate_fake_data(200)
        df_train = fake_data.drop("condition", axis=1)

        preprocessing_params = PreprocessingParams(
            use_scaler=True, use_custom_transformer=False, scaler="StandardScaler"
        )
        feature_params = FeatureParams(
            numerical_features=numerical_features(),
            categorical_features=categorical_features(),
            target_col=target_col(),
        )

        categorical_pipeline = build_categorical_pipeline(preprocessing_params)
        assert isinstance(categorical_pipeline, Pipeline)
        assert isinstance(categorical_pipeline["ohe"], OneHotEncoder)

        num_pipeline = build_numerical_pipeline(preprocessing_params)
        assert isinstance(num_pipeline, Pipeline)
        assert isinstance(num_pipeline["scaler"], StandardScaler)

        transformer = build_transformer(feature_params, preprocessing_params)
        assert isinstance(transformer, ColumnTransformer)

        transformer.fit(df_train)
        transform_data = process_data(transformer, df_train)
        assert transform_data.shape == (200, 22)


def test_custom_transformer(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        fake_data = generate_fake_data(200)
        df_train = fake_data.drop("condition", axis=1)
        df_train = df_train.drop(numerical_features(), axis=1)

        transformer = CategoricalTransformer()
        transform_data = transformer.fit_transform(df_train)
        print(transform_data)
        assert transform_data.shape == (200, 8)
        for col in categorical_features():
            assert 0 < transform_data[col].max() < 1
            assert 0 < transform_data[col].min() < 1

        transformer2 = CategoricalTransformer()
        transformer2.fit(df_train)
        transform_data = transformer2.transform(df_train)
        assert transform_data.shape == (200, 8)
        for col in categorical_features():
            assert 0 < transform_data[col].max() < 1
            assert 0 < transform_data[col].min() < 1


def test_predict(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        fake_data = generate_fake_data(200)
        target_train = fake_data["condition"]

        metrics = classifier_metrics(target_train.values, target_train)
        assert 1 >= metrics["accuracy"] >= 0
        assert 1 >= metrics["f1"] >= 0
        assert 1 >= metrics["precision"] >= 0
        assert 1 >= metrics["recall"] >= 0

        os.mkdir("models")
        predicts = save_predicts(target_train.values, "models/predicts.csv")
        assert os.path.exists(predicts)

        metrics_output = save_metrics(metrics, "models/metrics.json")
        assert os.path.exists(metrics_output)
