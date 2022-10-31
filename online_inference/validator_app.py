from typing import Literal
from pydantic import BaseModel, validator
from fastapi.exceptions import HTTPException


class MedicalResponse(BaseModel):
    id: int
    condition: Literal[0, 1]


class HeartDiseaseData(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]


    @validator('age')
    def validation_age(cls, v):
        if v < 0 or v > 120:
            raise HTTPException(detail=[{'msg':'ValueError: age value'}], status_code=400)
        return v


    @validator('trestbps')
    def validation_trestbps(cls, v):
        if v < 0 or v > 300:
            raise HTTPException(detail=[{'msg':'ValueError:wrong trestbps value'}], status_code=400)
        return v


    @validator('chol')
    def validation_chol(cls, v):
        if v < 0 or v > 600:
            raise HTTPException(detail=[{'msg':'ValueError: chol value'}], status_code=400)
        return v


    @validator('thalach')
    def validation_thalach(cls, v):
        if v < 0 or v > 300:
            raise HTTPException(detail=[{'msg':'ValueError: thalach value'}], status_code=400)
        return v


    @validator('oldpeak')
    def validation_oldpeak(cls, v):
        if v < 0 or v > 10:
            raise HTTPException(detail=[{'msg':'ValueError: oldpeak value'}], status_code=400)
        return v