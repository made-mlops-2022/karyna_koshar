import pathlib
import pandas as pd
import requests
import gdown
import json
import os


path_data = os.getenv("PATH_DATA", "data/heart_cleveland_upload.csv")
url = "https://drive.google.com/uc?export=download&id=1gOtgHlL-pm8wqQq8aYzYyoPoAZhGPMuo"
if not pathlib.Path(path_data).exists():
    os.mkdir("data")
    gdown.download(url=url, output=path_data, quiet=True)

data = pd.read_csv(path_data)
data.drop("condition", axis=1, inplace=True)

for row in data.to_dict(orient="records"):
    response = requests.post("http://0.0.0.0:5000/predict", json.dumps(row))
