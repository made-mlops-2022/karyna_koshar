import os
import pickle
import click
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier


@click.command("train")
@click.option("--input-dir", type=click.Path())
@click.option("--output-dir", type=click.Path())
def train(input_dir: str, output_dir: str) -> None:
    mlflow.set_tracking_uri("http://0.0.0.0:8000")
    with mlflow.start_run(run_name="train"):
        run_id = mlflow.active_run().info.run_id
        X = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
        y = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

        params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X, y)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="random-forest-classifier-model",
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"model_{run_id}.pkl"), "wb") as model_file:
            pickle.dump(model, model_file)


if __name__ == "__main__":
    train()
