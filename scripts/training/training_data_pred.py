import os
import pandas as pd
from prefect import task, flow

import warnings

warnings.filterwarnings("ignore")


train_path = "../data/bct-data-summit/train.csv"
test_path = "../data/bct-data-summit/test.csv"


@task(retries=3, retry_delay_seconds=2)
def load_data(path):

    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()
    return data


@task
def prepare_data(data):

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


@task
def save_data(data, save_path):

    if not os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
    data.to_csv(save_path, index=False)
    print("Data Successfully saved")


@flow
def main(data_path, save_path):

    data = load_data(data_path)
    data = prepare_data(data)
    save_data(data, save_path)


# if __name__ == "__main__":
#     main(train_path,'../data/newtrain2.csv')
