from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictParams:
    metric_path: str
    predicts_path: str
    model_path: str
    features_val_path: str
    target_val_path: str


PredictParamsSchema = class_schema(PredictParams)

def read_predict_params(path: str) -> PredictParams:
    with open(path, "r") as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
