import pickle
import pandas as pd

# from prefect import task, flow
from flask import Flask, jsonify, request


# @task
def load_model(model_path):

    with open(model_path, "rb") as f:
        model, scaler, vectorizer = pickle.load(f)
    return (model, scaler, vectorizer)


# @task
# def load_data(path):

#     data = pd.read_csv(path)
#     data.columns = data.columns.str.lower()
#     ids = data["id"].to_list()
#     rev_col = ["employeecount", "standardhours", "over18"]
#     data = data.drop(rev_col, axis=1)
#     return ids, data


def load_data(dicts):

    data = pd.DataFrame(dicts)
    data.columns = data.columns.str.lower()
    ids = data["id"].to_list()
    rev_col = ["employeecount", "standardhours", "over18"]
    data = data.drop(rev_col, axis=1)
    return ids, data


# @task
def data_prep(data, scaler, vectorizer):

    numerical_col = data.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_col = data.select_dtypes(include=["object"]).columns.tolist()

    data[numerical_col] = scaler.transform(data[numerical_col])
    data_dict = data[categorical_col + numerical_col].to_dict("records")
    x_values = vectorizer.transform(data_dict)

    return x_values


# @flow
def main(data_path, model_path):

    ids, data = load_data(data_path)
    model, scaler, vectorizer = load_model(model_path)
    encoded_data = data_prep(data, scaler, vectorizer)

    prediction = model.predict(encoded_data).round(2)
    prediction = [str(i) for i in prediction]
    dicts = {"id": ids, "Attrition": prediction}
    return dicts


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
# @flow
def predict():
    print(121212)
    data_path = request.get_json()
    print(1212)
    model_path = "./models/model.pkl"
    results = main(data_path, model_path)
    return jsonify(results)


if __name__ == "__main__":
    # main(data_path="../bct-data-summit/test.csv", model_path="../models/model.pkl")
    app.run(debug=True, port=5080)
