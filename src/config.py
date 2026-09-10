"""
Project configuration for the CDC BRFSS diabetes classification models.

This file stores shared constants used by the modelling pipeline, including
dataset paths, feature groups, train/validation/test proportions, cross-validation
settings, evaluation metrics, and hyperparameter search spaces.
"""

from pathlib import Path

from sklearn.metrics import make_scorer, precision_score

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# Reproducibility and dataset split
SEED = 97
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.125
CV_FOLDS = 5


# Target and feature groups
TARGET = "Diabetes_binary"
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
NOMINAL_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
QUANTITATIVE_FEATURES = ["BMI", "MentHlth", "PhysHlth"]
ALL_FEATURES = ORDINAL_FEATURES + NOMINAL_FEATURES + QUANTITATIVE_FEATURES
SCALED_FEATURES = ORDINAL_FEATURES + QUANTITATIVE_FEATURES


# Evaluation metrics
CV_SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Diabetes precision",
    "recall": "Diabetes recall",
    "f1": "Diabetes F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


# Decision tree hyperparameter search space
TREE_PARAM_GRID = {
    "model__max_depth": [3, 5, 7, 10],
    "model__min_samples_leaf": [50, 200, 500],
}


# Random forest hyperparameter search space
FOREST_PARAM_DISTRIBUTIONS = {
    "model__max_depth": [None, 10, 15, 20, 30],
    "model__min_samples_leaf": [1, 5, 10, 25, 50],
    "model__max_features": ["sqrt", 0.5, 0.75],
}
FOREST_SEARCH_ITERATIONS = 12


# XGBoost hyperparameter search space
XGBOOST_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [200, 400, 600],
    "model__learning_rate": [0.03, 0.05, 0.10],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_child_weight": [1, 5, 10],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_lambda": [1, 5, 10],
}
XGBOOST_SEARCH_ITERATIONS = 10