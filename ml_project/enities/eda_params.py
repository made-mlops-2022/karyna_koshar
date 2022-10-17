from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class EdaParams:
    path_with_data: str
    path_to_save: str


EdaParamsSchema = class_schema(EdaParams)

def read_eda_params(path: str) -> EdaParams:
    with open(path, "r") as input_stream:
        schema = EdaParamsSchema()
        return schema.load(yaml.safe_load(input_stream))