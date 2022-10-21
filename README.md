This project uses [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) dataset.

## Usage
This package allows you to train model for determine the heart disease.
1. Clone this repository to your machine.
2. Download [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) dataset, save csv locally (default path is *data* in repository's root).
3. If you want to get an EDA report with pandas profiling, the command should look like this. The generated report will be placed in *reports*.
```sh
python ml_project/eda.py configs/eda_config.yaml
```
There is also an EDA.ipynb report in *notebooks*.

4. Run train with the following command:
```sh
python ml_project/train_pipeline.py configs/train_config1.yaml
```
OR
```sh
python ml_project/train_pipeline.py configs/train_config2.yaml
```

4. Run predict with the following command:
```sh
python ml_project/predict.py configs/predict_config.yaml
```