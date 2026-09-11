"""
Overview page for the BRFSS Diabetes Analysis dashboard.

This page introduces the project, summarises the dataset, presents the modelling
workflow and highlights the main final results.
"""

import plotly.express as px
import streamlit as st

from utils import (
    TARGET, 
    load_data, 
    load_model_results
)


df = load_data()
_, validation_results, test_results, _ = load_model_results()

st.title("CDC BRFSS Diabetes Analysis")
st.caption("Exploratory Data Analysis | Statistical Testing | Imbalanced Classification")

dataset_url = "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data"
st.write(
    """
    This project explores health, lifestyle and demographic indicators associated
    with self-reported diabetes using a cleaned version of the [2015 CDC Behavioral Risk Factor
    Surveillance System (BRFSS) dataset](%s) published by Alex Teboul on Kaggle.
    """ 
    % dataset_url
)

st.info(
    "This project analyses survey-reported diabetes status and is not intended "
    "for clinical diagnosis or individual medical decision-making."
)

st.subheader("Dataset at a Glance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Survey Responses", f"{len(df):,}")
col2.metric("Predictors", "21")
col3.metric("Diabetes Prevalence", f"{df[TARGET].mean():.1%}")
col4.metric("Class Ratio", "6.18 : 1")

st.divider()

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Project Aim")

    st.write(
        """
        The analysis investigates relationships between diabetes status and
        health indicators before comparing classical machine learning models
        under substantial class imbalance.
        """
    )

    st.subheader("Workflow")

    st.markdown(
        """
        **1. Exploratory data analysis**  
        Investigate prevalence patterns across health, lifestyle and demographic variables.

        **2. Statistical analysis**  
        Quantify associations using appropriate statistical tests and effect sizes.

        **3. Machine learning**  
        Compare Logistic Regression, Decision Tree, Random Forest and XGBoost.

        **4. Class imbalance**  
        Compare standard models with class-weighted alternatives.

        **5. Evaluation**  
        Prioritise PR-AUC alongside balanced accuracy.
        """
    )

with right:
    st.subheader("Target Distribution")

    target_distribution = (
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename(index={0: "No Diabetes", 1: "Prediabetes & Diabetes"})
        .reset_index()
    )

    target_distribution.columns = ["Status", "Count"]
    target_distribution["Percentage"] = target_distribution["Count"] / len(df) * 100

    fig = px.bar(
        target_distribution,
        x="Status",
        y="Percentage",
        text="Percentage"
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=15, b=20),
        xaxis_title=None,
        yaxis_title="Respondents (%)",
        showlegend=False
    )

    st.plotly_chart(fig, width="stretch", height=330, config={"displayModeBar": False})

    st.caption(
        "The strong class imbalance means accuracy alone is not an appropriate "
        "measure of model quality."
    )

st.divider()
st.subheader("Final Model Performance")

final_result = test_results.iloc[0]

col1, col2, col3, col4 = st.columns([1.6, 1, 1, 1])

col1.metric("Selected Model", "Weighted XGBoost")
col2.metric("Test ROC-AUC", f"{final_result['ROC-AUC']:.3f}")
col3.metric("Test PR-AUC", f"{final_result['PR-AUC']:.3f}")
col4.metric("Diabetes Recall", f"{final_result['Diabetes recall']:.1%}")

st.write(
    """
    Weighted XGBoost produced the strongest overall discrimination. However, the gain over
    simpler models was relatively modest, showing that Logistic Regression already
    captured much of the predictive signal contained in the survey indicators.
    """
)

st.subheader("What This Project Demonstrates")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown(
            """
            **Data Analysis**
            - Exploratory data analysis
            - Health prevalence comparisons
            - Statistical hypothesis testing
            - Effect-size interpretation
            """
        )

with col2:
    with st.container(border=True):
        st.markdown(
            """
            **Machine Learning**
            - Stratified cross-validation
            - Hyperparameter tuning
            - Imbalanced classification
            - Precision-recall trade-offs
            """
        )