from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
from typing import Optional, List
import os
from validator_app import HeartDiseaseData, MedicalResponse

app = FastAPI()
model: Optional[Pipeline] = None


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as file_model:
        model = pickle.load(file_model)
        return model


@app.get("/")
def main():
    return "Main entrypoints..."


@app.on_event("startup")
def load_model():
    global model
    path_to_model = os.getenv("PATH_TO_MODEL", "model/model.pkl")
    model = load_object(path_to_model)


@app.get("/health")
def health() -> bool:
    return model is not None


@app.post("/predict", response_model=List[MedicalResponse])
def predict(data: HeartDiseaseData):
    input_df = pd.DataFrame([data.dict()])
    idx = list(i for i in range(input_df.shape[0]))
    predicts = model.predict(input_df)
    return list(
        MedicalResponse(id=i, condition=result) for i, result in zip(idx, predicts)
    )
