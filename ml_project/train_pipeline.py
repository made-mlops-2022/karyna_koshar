import json
import os
import sys
from pathlib import Path

import click
import pandas as pd

from enities.train_pipeline_params import read_train_pipeline_params

@click.command()
@click.argument("config_path")
def train_pipeline(config_path: str):
    train_pipeline_params = read_train_pipeline_params(config_path)

    return run_train_pipeline(train_pipeline_params)


def run_train_pipeline(train_pipeline_params):

    data = pd.read_csv(train_pipeline_params.input_data_path)
    print(data.shape)







if __name__ == "__main__":
    train_pipeline()