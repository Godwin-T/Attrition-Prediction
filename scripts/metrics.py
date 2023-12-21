import os
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from utils_and_constants import MODEL_PATH, MODEL_OUTPUT

if not os.path.exists("../model_output"):
    os.mkdir(MODEL_OUTPUT)


def plot_confusion_matrix(model, X_test, y_test):

    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig(f"{MODEL_OUTPUT}/predictions.csv")


def save_metrics(metrics):

    with open(f"{MODEL_OUTPUT}/metrics.json", "w") as fp:
        json.dump(metrics, fp)


def save_predictions(y_test, y_pred):
    # Store predictions data for confusion matrix
    cdf = pd.DataFrame(
        np.column_stack([y_test, y_pred]), columns=["true_label", "predicted_label"]
    ).astype(int)
    cdf.to_csv(f"{MODEL_OUTPUT}/predictions.csv", index=None)


def save_roc_curve(y_test, y_pred_proba):
    # Calcualte ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # Store roc curve data
    cdf = pd.DataFrame(np.column_stack([fpr, tpr]), columns=["fpr", "tpr"]).astype(
        float
    )
    cdf.to_csv(f"{MODEL_OUTPUT}/roc_curve.csv", index=None)


def save_model(model, scaler, vectorizer):

    with open(MODEL_PATH, "wb") as json_file:
        pickle.dump((model, scaler, vectorizer), json_file)
    print("Successfully saved")
