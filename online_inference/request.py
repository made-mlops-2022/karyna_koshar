import pandas as pd
import requests
import json
import os


path_data = os.getenv('PATH_DATA', 'data/heart_cleveland_upload.csv')
data = pd.read_csv(path_data)
data.drop('condition', axis=1, inplace=True)

for row in data.to_dict(orient='records'):
    response = requests.post(
        'http://0.0.0.0:5000/predict',
        json.dumps(row)
    )