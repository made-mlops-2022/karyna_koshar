from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainParams
from .preprocessing_params import PreprocessingParams


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    features_val_path: str
    metric_path: str
    splitting_params: SplittingParams
    preprocessing_params: PreprocessingParams
    feature_params: FeatureParams
    train_params: TrainParams
    use_mlflow: bool = field(default=False)


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
