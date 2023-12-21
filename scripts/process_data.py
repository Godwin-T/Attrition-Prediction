import os
from typing import List
import pandas as pd
from prefect import task, flow
from utils_and_constants import RAW_DATASET, DROP_COLNAMES, PROCESSED_DATASET

import warnings

warnings.filterwarnings("ignore")


# @task(retries=3, retry_delay_seconds=2)
def read_dataset(
    filename: str,
    drop_columns: List[str],
) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with Yes/No values

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    df = pd.read_csv(filename)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.drop(drop_columns, axis=1, inplace=True)

    categorical_cols = df.dtypes[df.dtypes == "object"].index.tolist()

    for col in categorical_cols:
        df[col] = df[col].str.lower().str.replace(" ", "_")
    return df


# @task
def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:

    data["newage"] = pd.cut(
        x=data["age"], bins=[17, 30, 42, 61], labels=["18 - 30", "31 - 42", "43 - 60"]
    )
    data["masterylevel"] = pd.cut(
        x=data["totalworkingyears"],
        bins=[-1, 3, 10, 421],
        labels=["entry", "intermediate", "master"],
    )
    data["loyaltylevel"] = pd.cut(
        x=data["yearsatcompany"],
        bins=[-1, 3, 10, 42],
        labels=["fairly", "loyal", "very-loyal"],
    )
    data["dueforprom"] = pd.cut(
        x=data["yearssincelastpromotion"], bins=[-1, 5, 16], labels=["due", "overdue"]
    )
    return data


# @flow
def main():

    dataframe = read_dataset(RAW_DATASET, drop_columns=DROP_COLNAMES)
    dataframe = prepare_dataset(dataframe)

    if not os.path.isdir("../processed_data"):
        os.mkdir("../processed_data")

    dataframe.to_csv(PROCESSED_DATASET, index=False)
    print("Data Successfully saved")


if __name__ == "__main__":
    main()
