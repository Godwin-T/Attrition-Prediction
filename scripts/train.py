import json
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from hyperopt.pyll import scope
from prefect import task, flow

import warnings

warnings.filterwarnings("ignore")
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN
from metrics import (
    save_metrics,
    save_roc_curve,
    save_predictions,
    save_model,
    plot_confusion_matrix,
)
from model import train_model, evaluate_model


# @task(retries=3, retry_delay_seconds=2)
def load_data(file_path):

    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def main():

    X, y = load_data(PROCESSED_DATASET)
    undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=1993)

    X, y = undersampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model, scaler, vectorizer, _ = train_model(X_train, y_train)
    test_metrics, y_proba, y_pred = evaluate_model(
        model, scaler, vectorizer, X_test, y_test
    )

    print("====================Test Set Metrics==================")
    print(json.dumps(test_metrics, indent=2))
    print("======================================================")

    save_metrics(test_metrics)
    save_roc_curve(y_test, y_proba)
    save_predictions(y_test, y_pred)
    save_model(model, scaler, vectorizer)


if __name__ == "__main__":
    main()
