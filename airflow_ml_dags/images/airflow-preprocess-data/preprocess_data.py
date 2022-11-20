import os
import click
import numpy as np
import pandas as pd


@click.command("preprocess_data")
@click.option("--input-dir", type=click.Path())
@click.option("--output-dir", type=click.Path())
def preprocess_data(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    data["oldpeak"] = np.round(data["oldpeak"], decimals=2)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    preprocess_data()
