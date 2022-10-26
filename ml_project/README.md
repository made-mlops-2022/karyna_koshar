# MADE_MLOps_Homework_1

This project uses [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) dataset.

## Usage
This package allows you to train model for determine the heart disease.
1. Clone this repository to your machine.
2. Download [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) dataset, save csv locally (default path is *data* in repository's root).
3. Install dependencies:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
4. If you want to get an EDA report with pandas profiling, the command should look like this. The generated report will be placed in *reports*.
```sh
python ml_project/eda.py configs/eda_config.yaml
```
There is also an EDA.ipynb report in *notebooks*.

5. Run train with the following command:
```sh
python ml_project/train_pipeline.py configs/train_config_forest.yaml
```
OR
```sh
python ml_project/train_pipeline.py configs/train_config_logreg.yaml
```

6. Run predict with the following command:
```sh
python ml_project/predict.py configs/predict_config.yaml
```
7. Run tests:
```sh
pytest tests/
```
8. Run MLflow UI to see the information about experiments you conducted:
```sh
mlflow ui
```
![mlflow logs](https://user-images.githubusercontent.com/98235486/197872052-b56c043f-0d37-4e46-9766-e7db3a9f48eb.PNG)
