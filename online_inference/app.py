from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
from typing import Optional, List, Literal
from pydantic import BaseModel
import os
import uvicorn

app = FastAPI()
model: Optional[Pipeline] = None


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


class MedicalResponse(BaseModel):
    id: int
    condition: Literal[0, 1]


def load_object(path: str) -> Pipeline:
    with open(path, 'rb') as file_model:
        model = pickle.load(file_model)
        return model


@app.get('/')
def main():
    return 'Main entrypoints...'


@app.on_event('startup')
def load_model():
    global model
    path_to_model = os.getenv('PATH_TO_MODEL', 'model/model.pkl')
    model = load_object(path_to_model)


@app.get('/health')
def health() -> bool:
    return model is not None


@app.post('/predict', response_model=List[MedicalResponse])
def predict(data: HeartDiseaseData):
    input_df = pd.DataFrame([data.dict()])
    idx = list(i for i in range(input_df.shape[0]))
    predicts = model.predict(input_df)
    return list(MedicalResponse(id=i, condition=result) for i, result in zip(idx, predicts))
