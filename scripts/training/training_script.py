import pickle
import pandas as pd

import mlflow
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from imblearn.under_sampling import RandomUnderSampler

from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import task, flow

import warnings

warnings.filterwarnings("ignore")


@task(retries=3, retry_delay_seconds=2)
def load_data(path):

    data = pd.read_csv(path)
    rev_col = ["id", "employeecount", "standardhours", "over18"]
    data = data.drop(rev_col, axis=1)
    return data


@task
def split_data(data, random_state):

    train_df, test_df = train_test_split(
        data, test_size=0.25, random_state=random_state
    )
    train_y, test_y = train_df.pop("attrition"), test_df.pop("attrition")
    output = (train_df, train_y, test_df, test_y)
    return output


@task
def data_sampling(data, random_state):

    undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=random_state)
    y = data.pop("attrition")
    train_x, y = undersampler.fit_resample(data, y)
    train_x["attrition"] = y
    data = train_x
    return data


@task
def vectorise_data(train_df, test_df):

    numerical_col = train_df.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_col = train_df.select_dtypes(include=["object"]).columns.tolist()

    vectorizer = DictVectorizer()
    train_dicts = train_df[categorical_col + numerical_col].to_dict(orient="records")
    val_dicts = test_df[categorical_col + numerical_col].to_dict(orient="records")

    vectorizer.fit(train_dicts)
    X_train = vectorizer.transform(train_dicts)
    X_val = vectorizer.transform(val_dicts)
    output = (X_train, X_val, vectorizer)
    return output


@task
def hyperparameter_search(dtrain, dtest, X_val, test_y):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "lgb")
            mlflow.set_tag("data", "engineered")
            mlflow.set_tag("loss", "f1")
            mlflow.log_params(params)

            # (dtrain, dtest, X_val, test_y, vectorizer) = prepare_data(data_path)
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=1000,
                valid_sets=[dtest],
            )
            prediction0 = booster.predict(X_val)
            prediction = (prediction0 >= 0.5).astype("int")
            aroc = roc_auc_score(test_y, prediction0)
            output = {
                "acc": accuracy_score(test_y, prediction),
                "f1_score": f1_score(test_y, prediction),
                "precision": precision_score(test_y, prediction),
                "recall": recall_score(test_y, prediction),
                "area_roc": aroc,
            }
            mlflow.log_metrics(output)

        return {"loss": -aroc, "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "boosting": "gbdt",
        "num_iterations": 120,
        "num_leaves": scope.int(hp.quniform("num_leaves", 4, 100, 1)),
        "seed": 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )
    print(best_result)
    print("Successfully Completed")
    return best_result


@task
def train_best_params(dtrain, dtest, X_val, test_y, best_result):

    with mlflow.start_run():
        mlflow.set_tag("model", "lgb")
        mlflow.set_tag("params", "best")

        best_params = {
            "max_depth": int(best_result["max_depth"]),
            "learning_rate": best_result["learning_rate"],
            "boosting": "gbdt",
            "num_iteration": 120,
            "num_leaves": int(best_result["num_leaves"]),
        }

        booster = lgb.train(
            params=best_params,
            train_set=dtrain,
            num_boost_round=1000,
            valid_sets=[dtest],
        )

        prediction0 = booster.predict(X_val)
        prediction = (prediction0 >= 0.5).astype("int")
        f1 = f1_score(test_y, prediction)
        output = {
            "acc": accuracy_score(test_y, prediction),
            "f1_score": f1,
            "precision": precision_score(test_y, prediction),
            "recall": recall_score(test_y, prediction),
            "area_roc": roc_auc_score(test_y, prediction0),
        }
        mlflow.log_params(best_params)
        mlflow.log_metrics(output)
        return booster


@task
def save_model(model, vectorizer, save_path):

    model_uri = save_path
    with open(model_uri, "wb") as f:
        pickle.dump((model, vectorizer), f)

    mlflow.log_artifact(save_path, "model")
    mlflow.pyfunc.log_model(
        "model", data_path=save_path, loader_module="mlflow.sklearn"
    )
    print("Successfully saved")


# @flow
# def prepare_data(data_path, random_state = 0):

#     data = load_data(data_path)
#     data = data_sampling(data, random_state)

#     train_df, train_y, test_df, test_y = split_data(data, random_state)
#     X_train, X_val, vectorizer = vectorise_data(train_df, test_df)

#     dtrain = lgb.Dataset(X_train, label = train_y, free_raw_data=False)
#     dtest = lgb.Dataset(X_val, label = test_y, reference=dtrain,
#                         free_raw_data=False)
#     output = (dtrain, dtest, X_val,test_y, vectorizer)
#     return output


@flow
def main(
    datapath="../../data/newtrain2.csv",
    save_path="../../models/model.pkl",
    random_state=0,
):

    mlflow.set_tracking_uri("sqlite:///../../notebooks/mlflow.db")
    mlflow.set_experiment("Attrition")

    data = load_data(datapath)
    data = data_sampling(data, random_state)

    train_df, train_y, test_df, test_y = split_data(data, random_state)
    X_train, X_val, vectorizer = vectorise_data(train_df, test_df)

    dtrain = lgb.Dataset(X_train, label=train_y, free_raw_data=False)
    dtest = lgb.Dataset(X_val, label=test_y, reference=dtrain, free_raw_data=False)

    best_result_params = hyperparameter_search(dtrain, dtest, X_val, test_y)
    model = train_best_params(dtrain, dtest, X_val, test_y, best_result_params)
    save_model(model, vectorizer, save_path)


if __name__ == "__main__":
    main()
