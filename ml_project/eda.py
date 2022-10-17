import pandas as pd
from pandas_profiling import ProfileReport
from enities.eda_params import read_eda_params
import click


@click.command()
@click.argument("config_path")
def eda_pandas_profiling(config_path: str) -> None:
    eda_params = read_eda_params(config_path)
    dataset = pd.read_csv(eda_params.path_with_data)
    profile = ProfileReport(dataset, title="Pandas Profiling Report", explorative=True)
    profile.to_file(eda_params.path_to_save)


if __name__ == "__main__":
    eda_pandas_profiling()
