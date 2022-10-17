from dataclasses import dataclass, field


@dataclass()
class PreprocessingParams:
    use_scaler: bool = field(default=True)
    scaler: str = field(default="StandartScaler")
