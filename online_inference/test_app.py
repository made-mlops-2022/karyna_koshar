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
        'age': 40,
        'sex': 1,
        'cp': 0,
        'trestbps': 140,
        'chol': 199,
        'fbs': 0,
        'restecg': 0,
        'thalach': 178,
        'exang': 1,
        'oldpeak': 1.4,
        'slope': 0,
        'ca': 0,
        'thal': 2
        }
    response = client.post('/predict', json.dumps(request))
    assert response.status_code == 200
    assert response.json() == [{'condition': 0, 'id': 0}]


def test_predict_categorical_422():
    request = {
        'age': 40,
        'sex': 3,
        'cp': 0,
        'trestbps': 140,
        'chol': 199,
        'fbs': 0,
        'restecg': 0,
        'thalach': 178,
        'exang': 1,
        'oldpeak': 1.4,
        'slope': 0,
        'ca': 0,
        'thal': 2
        }
    response = client.post('/predict', json.dumps(request))
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1'


def test_predict_numerical_400():
    request = {
        'age': 40,
        'sex': 1,
        'cp': 0,
        'trestbps': 140,
        'chol': 5000,
        'fbs': 0,
        'restecg': 0,
        'thalach': 178,
        'exang': 1,
        'oldpeak': 1.4,
        'slope': 0,
        'ca': 0,
        'thal': 2
        }
    response = client.post('/predict', json.dumps(request))
    assert response.status_code == 400
    assert response.json()['detail'][0]['msg'] == 'ValueError: chol value'


def test_predict_field_skipped_422():
    request = {
        'sex': 1,
        'cp': 0,
        'trestbps': 140,
        'chol': 199,
        'fbs': 0,
        'restecg': 0,
        'thalach': 178,
        'exang': 1,
        'oldpeak': 1.4,
        'slope': 0,
        'ca': 0,
        'thal': 2
        }
    response = client.post('/predict', json.dumps(request))
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'
