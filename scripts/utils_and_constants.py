import os
import shutil
from pathlib import Path

if os.getenv("DVC"):
    DATASET_TYPES = ["test", "train"]
    DROP_COLNAMES = ["employeecount", "standardhours", "over18"]
    TARGET_COLUMN = "attrition"
    RAW_DATASET = "./bct-data-summit/train.csv"
    PROCESSED_DATASET = "./processed_data/attrition.csv"
    PARAMETERS = "./parameters.json"
    MODEL_PATH = "./models/model.pkl"
    MODEL_OUTPUT = "./model_output"
else:
    DATASET_TYPES = ["test", "train"]
    DROP_COLNAMES = ["employeecount", "standardhours", "over18"]
    TARGET_COLUMN = "attrition"
    RAW_DATASET = "../bct-data-summit/train.csv"
    PROCESSED_DATASET = "../processed_data/attrition.csv"
    MODEL_PATH = "../models/model.pkl"
    PARAMETERS = "../parameters.json"
    MODEL_OUTPUT = "../model_output"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)
