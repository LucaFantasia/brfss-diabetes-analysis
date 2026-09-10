"""
Evaluation methods for the CDC BRFSS diabetes classification models.

This module contains reusable functions for evaluating classifiers, extracting
cross-validation results, and saving experiment outputs.
"""

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def get_positive_class_probability(model, X):
    """
    Return the predicted probability for the positive diabetes class.
    """

    positive_class_index = np.where(model.classes_ == 1)[0][0]

    return model.predict_proba(X)[:, positive_class_index]


def evaluate_classifier(model_name, y_true, y_pred, y_probability):
    """
    Calculate classification metrics for a fitted model.

    Accuracy is reported for completeness, but balanced accuracy, precision,
    recall, F1, ROC-AUC, and PR-AUC are particularly important because the
    diabetes target is heavily imbalanced.
    """

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Diabetes precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Diabetes recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Diabetes F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_probability),
        "PR-AUC": average_precision_score(y_true, y_probability)
    }


def extract_cross_validate_results(model_name, cv_results, metric_labels):
    """
    Convert scikit-learn cross_validate output into a single result row.
    """

    result = {
        "Model": model_name
    }

    for metric_name, display_name in metric_labels.items():
        scores = cv_results[f"test_{metric_name}"]
        result[f"{display_name} mean"] = scores.mean()
        result[f"{display_name} std"] = scores.std()

    return result


def extract_search_results(model_name, search, metric_labels):
    """
    Extract cross-validation metrics for the best hyperparameter configuration.

    The best configuration is selected by the PR-AUC metric.
    """

    result = {
        "Model": model_name
    }

    best_index = search.best_index_

    for metric_name, display_name in metric_labels.items():
        result[f"{display_name} mean"] = search.cv_results_[f"mean_test_{metric_name}"][best_index]
        result[f"{display_name} std"] = search.cv_results_[f"std_test_{metric_name}"][best_index]

    return result


def make_json_safe(value):
    """
    Recursively convert NumPy values into JSON-compatible Python objects.
    """

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def save_results(dataframe, path):
    """
    Save a pandas DataFrame as a CSV file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(path, index=False)


def save_json(data, path):
    """
    Save experiment metadata as a formatted JSON file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=4)