import pickle
import pandas as pd
from prefect import task, flow
from flask import Flask, jsonify, request


@task
def load_model(model_path):

    with open(model_path, "rb") as f:
        model, vectorizer = pickle.load(f)
    return (model, vectorizer)


@task
def load_data(path):

    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()
    ids = data["id"].to_list()
    rev_col = ["id", "employeecount", "standardhours", "over18"]
    data = data.drop(rev_col, axis=1)
    return ids, data


@task
def feature_engineering(data):

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
def data_prep(data, vectorizer):

    numerical_col = data.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_col = data.select_dtypes(include=["object"]).columns.tolist()

    data_dict = data[categorical_col + numerical_col].to_dict(orient="record")
    x_values = vectorizer.transform(data_dict)
    return x_values


@flow
def main(data_path, model_path):

    model, vectorizer = load_model(model_path)
    ids, data = load_data(data_path)
    data = feature_engineering(data)
    encoded_data = data_prep(data, vectorizer)

    prediction = model.predict(encoded_data).round(2)
    dicts = {"id": ids, "Attrition": prediction}
    output_frame = pd.DataFrame(dicts)
    output_frame.to_csv("../../Attrition.csv", index=False)
    return {"Attrition_path": "../../Attrition.csv"}


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
@flow
def predict():

    data_path = request.get_json()
    model_path = "../../models/model.pkl"
    results = main(data_path, model_path)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5080)
