import json
from fastapi.testclient import TestClient
from app import app, load_model

client = TestClient(app)


def test_health_200():
    load_model()
    response = client.get('/health')
    assert response.status_code == 200


def test_predict_disease_200():
    request = {
        'age': 64,
        'sex': 1,
        'cp': 0,
        'trestbps': 170,
        'chol': 277,
        'fbs': 0,
        'restecg': 2,
        'thalach': 155,
        'exang': 0,
        'oldpeak': 0.6,
        'slope': 1,
        'ca': 0,
        'thal': 2
        }
    response = client.post('/predict', json.dumps(request))
    assert response.status_code == 200
    assert response.json() == [{'condition': 1, 'id': 0}]


def test_predict_no_disease_200():
    request = {
        'age': 59,
        'sex': 1,
        'cp': 0,
        'trestbps': 134,
        'chol': 204,
        'fbs': 0,
        'restecg': 0,
        'thalach': 162,
        'exang': 0,
        'oldpeak': 0.8,
        'slope': 0,
        'ca': 2,
        'thal': 0
        }
    response = client.post('/predict', json.dumps(request))
    assert response.status_code == 200
    assert response.json() == [{'condition': 1, 'id': 0}]