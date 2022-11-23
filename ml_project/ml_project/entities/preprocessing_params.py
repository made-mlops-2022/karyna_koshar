from dataclasses import dataclass, field


@dataclass()
class PreprocessingParams:
    use_scaler: bool = field(default=True)
    use_custom_transformer: bool = field(default=False)
    scaler: str = field(default="StandartScaler")
