from fastapi import FastAPI, Request
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

api = FastAPI()

path = "model.pkl"
def load_object(path: str) -> Pipeline:
    with open(path, 'rb') as file_model:
        model = pickle.load(file_model)
        return model

