from typing import List
from faker import Faker
import pandas as pd

from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.split_params import SplittingParams
from ml_project.enities.preprocessing_params import PreprocessingParams
from ml_project.enities.train_params import TrainParams
from ml_project.enities.train_pipeline_params import TrainPipelineParams
from ml_project.enities.predict_params import PredictParams


def generate_fake_data(num_rows: int) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(42)
    output = [
        {
            "sex": fake.random_int(min=0, max=1),
            "cp": fake.random_int(min=0, max=4),
            "fbs": fake.random_int(min=0, max=1),
            "restecg": fake.random_int(min=0, max=2),
            "exang": fake.random_int(min=0, max=1),
            "slope": fake.random_int(min=0, max=2),
            "ca": fake.random_int(min=0, max=4),
            "thal": fake.random_int(min=0, max=2),
            "age": fake.random_int(min=29, max=77),
            "trestbps": fake.random_int(min=94, max=200),
            "chol": fake.random_int(min=126, max=564),
            "thalach": fake.random_int(min=71, max=202),
            "oldpeak": fake.pyfloat(min_value=0, max_value=6.2),
            "condition": fake.random_int(min=0, max=1),
        }
        for x in range(num_rows)
    ]

    return pd.DataFrame(output, columns=get_columns())


def target_col() -> str:
    return "condition"


def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


def get_columns() -> list[str]:
    return [target_col()] + categorical_features() + numerical_features()


def create_config_train() -> TrainPipelineParams:
    params = TrainPipelineParams(
        input_data_path="data/heart_cleveland_upload.csv",
        output_model_path="models/model.pkl",
        features_val_path="data/features_val.csv",
        target_val_path="data/target_val.csv",
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        preprocessing_params=PreprocessingParams(
            use_scaler=True, use_custom_transformer=False, scaler="StandardScaler"
        ),
        feature_params=FeatureParams(
            numerical_features=numerical_features(),
            categorical_features=categorical_features(),
            target_col=target_col(),
        ),
        train_params=TrainParams(model_type="LogisticRegression"),
    )
    return params


def create_config_predict() -> PredictParams:
    params = PredictParams(
        metric_path="models/metrics.json",
        predicts_path="models/predicts.csv",
        model_path="models/model.pkl",
        features_val_path="data/features_val.csv",
        target_val_path="data/target_val.csv",
        use_mlflow=False,
    )
    return params
