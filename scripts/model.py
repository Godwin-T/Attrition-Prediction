import json
import pickle

import lightgbm as lgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from prefect import task, flow

import warnings

warnings.filterwarnings("ignore")
from utils_and_constants import PARAMETERS, MODEL_PATH


def prep_data(X_train):

    categorical_col = X_train.dtypes[X_train.dtypes == "object"].index.tolist()
    numerical_col = X_train.dtypes[X_train.dtypes != "object"].index.tolist()

    scaler = StandardScaler()
    print(numerical_col, categorical_col)
    print(X_train[numerical_col].shape)
    X_train[numerical_col] = scaler.fit_transform(X_train[numerical_col])

    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit(X_train[categorical_col + numerical_col].to_dict(orient="records"))
    X_train = vectorizer.transform(
        X_train[categorical_col + numerical_col].to_dict(orient="records")
    )
    return X_train, scaler, vectorizer


def eval_metrics(y_true, prediction):

    f1 = f1_score(y_true, prediction)
    metrics = {
        "acc": accuracy_score(y_true, prediction),
        "f1_score": f1,
        "precision": precision_score(y_true, prediction),
        "recall": recall_score(y_true, prediction),
    }
    return metrics


def train_model(X_train, y_train):

    with open(PARAMETERS, "r") as json_file:
        parameters = json.load(json_file)
        parameters["max_depth"] = int(parameters["max_depth"])
        parameters["num_leaves"] = int(parameters["num_leaves"])

    X_train, scaler, vectorizer = prep_data(X_train)
    X_train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    booster = lgb.train(
        params=parameters,
        train_set=X_train_dataset,
        num_boost_round=1000,
        valid_sets=[X_train_dataset],
    )

    prediction0 = booster.predict(X_train)
    prediction = (prediction0 >= 0.5).astype("int")
    metrics = eval_metrics(y_train, prediction)

    return booster, scaler, vectorizer, metrics


def evaluate_model(model, scaler, vectorizer, X_test, y_test, float_precision=4):

    categorical_col = X_test.dtypes[X_test.dtypes == "object"].index.tolist()
    numerical_col = X_test.dtypes[X_test.dtypes != "object"].index.tolist()

    X_test[numerical_col] = scaler.transform(X_test[numerical_col])
    X_test = vectorizer.transform(
        (X_test[categorical_col + numerical_col].to_dict(orient="records"))
    )

    y_proba = model.predict(X_test)
    prediction = (y_proba >= 0.5).astype("int")
    metrics = eval_metrics(y_test, prediction)

    metrics = json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precision)
    )
    return metrics, y_proba, prediction


def save_model(model, scaler, vectorizer):

    with open(MODEL_PATH, "wb") as f:
        pickle.dump([model, scaler, vectorizer], f)
    print("Model saved successfully!")
