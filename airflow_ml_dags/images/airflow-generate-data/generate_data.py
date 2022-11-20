from typing import List
from faker import Faker
import pandas as pd
import click
import os


@click.command("generate_data")
@click.option("--output-dir", type=click.Path())
def generate_data(output_dir: str) -> None:
    num_rows = 250
    fake = Faker()
    Faker.seed(42)
    output = [
        {
            "sex": fake.random_int(min=0, max=1),
            "cp": fake.random_int(min=0, max=4),
            "fbs": fake.random_int(min=0, max=1),
            "restecg": fake.random_int(min=0, max=2),
            "exang": fake.random_int(min=0, max=1),
            "slope": fake.random_int(min=0, max=2),
            "ca": fake.random_int(min=0, max=4),
            "thal": fake.random_int(min=0, max=2),
            "age": fake.random_int(min=29, max=77),
            "trestbps": fake.random_int(min=94, max=200),
            "chol": fake.random_int(min=126, max=564),
            "thalach": fake.random_int(min=71, max=202),
            "oldpeak": fake.pyfloat(min_value=0, max_value=6.2),
            "condition": fake.random_int(min=0, max=1),
        }
        for _ in range(num_rows)
    ]

    os.makedirs(output_dir, exist_ok=True)
    data = pd.DataFrame(output, columns=get_columns())
    X = data.drop(target_col(), axis=1)
    y = data[target_col()]
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


def target_col() -> str:
    return "condition"


def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


def get_columns() -> List[str]:
    return [target_col()] + categorical_features() + numerical_features()


if __name__ == "__main__":
    generate_data()
