# MADE_MLOps_Homework_3

This project uses [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) dataset.

## Usage
1. If you want to start airflow, run this script:
```sh
./start.sh
```
Available at http://localhost:8080 (login: admin, password: admin).

2. Run tests:
```sh
docker exec -it airflow_ml_dags-scheduler-1 bash
pip3 install pytest
python3 -m pytest tests/
```