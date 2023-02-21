import re

import numpy as np
import pandas as pd


def _get_first_cabin(row):
    try:
        return row.split()[0]
    except Exception:
        return np.nan


def _get_title(passenger) -> str:
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"

def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    data_copy = dataframe.copy()

    data_copy = data_copy.replace("?", np.nan)
    data_copy["cabin"] = data_copy["cabin"].apply(_get_first_cabin)
    data_copy["title"] = data_copy["name"].apply(_get_title)
    data_copy["fare"] = data_copy["fare"].astype("float")
    data_copy["age"] = data_copy["age"].astype("float")
    data_copy.drop(
        labels=["name", "ticket", "boat", "body", "home.dest"], axis=1, inplace=True
    )

    return data_copy