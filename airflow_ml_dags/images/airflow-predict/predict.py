import os
import click
import pandas as pd
import mlflow


@click.command("predict")
@click.option("--input-dir", type=click.Path())
@click.option("--output-dir", type=click.Path())
def predict(input_dir: str, output_dir: str) -> None:
    X = pd.read_csv(os.path.join(input_dir, "train_data.csv"))

    mlflow.set_tracking_uri("http://0.0.0.0:8000")
    model = mlflow.pyfunc.load_model(
        model_uri="models:/random-forest-classifier-model/Production"
    )

    os.makedirs(output_dir, exist_ok=True)
    predicts = model.predict(X)
    predicts = pd.DataFrame(predicts, columns=["predicts"])
    predicts.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    predict()
