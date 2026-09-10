"""
Shared utilities for the BRFSS Diabetes Analysis Streamlit dashboard.

This module stores project paths, feature metadata, data-loading functions,
statistical calculations and prevalence helpers shared between dashboard pages.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET = "Diabetes_binary"
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
NOMINAL_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex"
]
QUANTITATIVE_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

FEATURE_LABELS = {
    "GenHlth": "General Health",
    "Age": "Age",
    "Education": "Education",
    "Income": "Income",
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check",
    "Smoker": "Smoker",
    "Stroke": "Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Heart Attack",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Consumption",
    "Veggies": "Vegetable Consumption",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Unable to See Doctor Due to Cost",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex",
    "BMI": "BMI",
    "MentHlth": "Poor Mental Health Days",
    "PhysHlth": "Poor Physical Health Days"
}
ORDINAL_LABELS = {
    "GenHlth": {
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    },
    "Age": {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60-64",
        10: "65-69",
        11: "70-74",
        12: "75-79",
        13: "80+"
    },
    "Education": {
        1: "None / Kindergarten",
        2: "Grades 1-8",
        3: "Grades 9-11",
        4: "High School / GED",
        5: "Some College",
        6: "College Graduate"
    },
    "Income": {
        1: "<$10k",
        2: "$10k-<$15k",
        3: "$15k-<$20k",
        4: "$20k-<$25k",
        5: "$25k-<$35k",
        6: "$35k-<$50k",
        7: "$50k-<$75k",
        8: "$75k+"
    }
}

EDA_INSIGHTS = {
    "GenHlth": "Diabetes prevalence increases substantially as self-reported general health worsens.",
    "Age": "Diabetes prevalence is generally higher among older respondents.",
    "Education": "Higher education categories are associated with lower observed diabetes prevalence.",
    "Income": "Higher income categories are associated with lower observed diabetes prevalence.",
    "BMI": "Diabetes prevalence increases clearly across higher BMI ranges.",
    "PhysHlth": "Respondents reporting more physically unhealthy days generally show higher diabetes prevalence.",
    "MentHlth": "Mental-health days show a weaker relationship with diabetes than BMI or physical health."
}


@st.cache_data
def load_data():
    """
    Load the CDC BRFSS diabetes dataset from the UCI Machine Learning Repository.
    """
    dataset = fetch_ucirepo(id=891)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = X.copy()
    df[TARGET] = y[TARGET]

    return df


@st.cache_data
def load_model_results():
    """
    Load saved model evaluation results and experiment metadata.
    """
    cv_results = pd.read_csv(OUTPUT_DIR / "cv_results.csv")
    validation_results = pd.read_csv(OUTPUT_DIR / "validation_results.csv")
    test_results = pd.read_csv(OUTPUT_DIR / "test_results.csv")

    with open(OUTPUT_DIR / "best_params.json", "r", encoding="utf-8") as file:
        best_params = json.load(file)

    return cv_results, validation_results, test_results, best_params


def format_p_value(value):
    """
    Format very small p-values without displaying them as zero.
    """
    if value == 0 or value < 1e-300:
        return "<1e-300"

    if value < 0.001:
        return f"{value:.2e}"

    return f"{value:.3f}"


def ordinal_prevalence(df, feature):
    """
    Calculate diabetes prevalence across an ordinal feature.
    """
    summary = df.groupby(feature, observed=False)[TARGET].agg(["mean", "size"]).reset_index()
    summary["Diabetes prevalence"] = summary["mean"] * 100
    summary["Category"] = summary[feature].map(ORDINAL_LABELS[feature])

    return summary


def binary_prevalence(df, feature):
    """
    Calculate diabetes prevalence for the two values of a binary feature.
    """
    summary = df.groupby(feature)[TARGET].agg(["mean", "size"]).reset_index()
    summary["Diabetes prevalence"] = summary["mean"] * 100

    if feature == "Sex":
        summary["Category"] = summary[feature].map({0: "Female", 1: "Male"})
    else:
        summary["Category"] = summary[feature].map({0: "No", 1: "Yes"})

    return summary


def quantitative_prevalence(df, feature):
    """
    Bin quantitative health features into readable ranges and calculate prevalence.
    """
    if feature == "BMI":
        bins = [-np.inf, 19, 24, 29, 34, 39, np.inf]
        labels = ["≤19", "20–24", "25–29", "30–34", "35–39", "40+"]
    else:
        bins = [-1, 0, 5, 10, 20, 29, 30]
        labels = ["0", "1–5", "6–10", "11–20", "21–29", "30"]

    groups = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)
    summary = df.groupby(groups, observed=False)[TARGET].agg(["mean", "size"]).reset_index()
    summary["Diabetes prevalence"] = summary["mean"] * 100
    summary["Category"] = summary[feature].astype(str)

    return summary


@st.cache_data
def calculate_statistical_results(df):
    """
    Calculate statistical tests and effect sizes for all three feature groups.
    """
    binary_results = []

    for feature in NOMINAL_FEATURES:
        table = pd.crosstab(df[feature], df[TARGET])
        chi2, p_value, _, _ = chi2_contingency(table, correction=False)

        n = table.to_numpy().sum()
        cramers_v = np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1)))

        binary_results.append({
            "Feature": FEATURE_LABELS[feature],
            "Cramér's V": cramers_v,
            "p-value": p_value
        })

    binary_results = pd.DataFrame(binary_results)
    binary_results["Adjusted p-value"] = multipletests(binary_results["p-value"], method="fdr_bh")[1]
    binary_results = binary_results.sort_values("Cramér's V", ascending=False).reset_index(drop=True)

    ordinal_results = []

    for feature in ORDINAL_FEATURES:
        correlation, p_value = spearmanr(df[feature], df[TARGET])

        ordinal_results.append({
            "Feature": FEATURE_LABELS[feature],
            "Spearman rho": correlation,
            "p-value": p_value
        })

    ordinal_results = pd.DataFrame(ordinal_results)
    ordinal_results["Adjusted p-value"] = multipletests(ordinal_results["p-value"], method="fdr_bh")[1]
    ordinal_results["Absolute rho"] = ordinal_results["Spearman rho"].abs()
    ordinal_results = ordinal_results.sort_values("Absolute rho", ascending=False)
    ordinal_results = ordinal_results.drop(columns="Absolute rho").reset_index(drop=True)

    no_diabetes = df[df[TARGET] == 0]
    diabetes = df[df[TARGET] == 1]

    quantitative_results = []

    for feature in QUANTITATIVE_FEATURES:
        group_0 = no_diabetes[feature]
        group_1 = diabetes[feature]

        u_statistic, p_value = mannwhitneyu(group_1, group_0, alternative="two-sided")
        rank_biserial = 2 * u_statistic / (len(group_1) * len(group_0)) - 1

        quantitative_results.append({
            "Feature": FEATURE_LABELS[feature],
            "Median: no diabetes": group_0.median(),
            "Median: diabetes": group_1.median(),
            "Rank-biserial": rank_biserial,
            "p-value": p_value
        })

    quantitative_results = pd.DataFrame(quantitative_results)
    quantitative_results["Adjusted p-value"] = multipletests(
        quantitative_results["p-value"],
        method="fdr_bh"
    )[1]

    quantitative_results["Absolute effect"] = quantitative_results["Rank-biserial"].abs()
    quantitative_results = quantitative_results.sort_values("Absolute effect", ascending=False)
    quantitative_results = quantitative_results.drop(columns="Absolute effect").reset_index(drop=True)

    return binary_results, ordinal_results, quantitative_results