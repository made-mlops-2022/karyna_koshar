import os
import pandas as pd
from pandas_profiling import ProfileReport


def eda_pandas_profiling() -> None:
    path_with_data = "data/heart_cleveland_upload.csv"
    path_to_save = "reports"
    dataset = pd.read_csv(path_with_data)
    profile = ProfileReport(dataset, title="Pandas Profiling Report", explorative=True)
    profile.to_file(os.path.join(path_to_save, "EDA.html"))


eda_pandas_profiling()
