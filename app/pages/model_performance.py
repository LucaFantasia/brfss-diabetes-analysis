"""
Model Performance page for the BRFSS Diabetes Analysis dashboard.

This page compares cross-validation and validation performance across the
classification models, examines class weighting and presents final test results.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_model_results


cv_results, validation_results, test_results, best_params = load_model_results()

st.title("Model Performance")
st.caption("Comparing classical classifiers under substantial class imbalance.")

st.write(
    """
    Logistic Regression, Decision Tree, Random Forest and XGBoost models were
    evaluated using a stratified 70/10/20 train/validation/test split respectively. Five-fold
    stratified cross-validation was used on the training set, with PR-AUC and balanced accuracy used as
    the primary and secondary model-selection metric respectively.
    """
)

final_result = test_results.iloc[0]

col1, col2, col3, col4 = st.columns([1.6, 1, 1, 1])

col1.metric("Final Model", "Weighted XGBoost")
col2.metric("Test ROC-AUC", f"{final_result['ROC-AUC']:.3f}")
col3.metric("Test PR-AUC", f"{final_result['PR-AUC']:.3f}")
col4.metric("Balanced Accuracy", f"{final_result['Balanced accuracy']:.3f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Cross-Validation",
    "Validation Comparison",
    "Class Weighting",
    "Final Test"
])

with tab1:
    st.subheader("Cross-Validation Performance")

    metric = st.selectbox(
        "Cross-validation metric",
        ["PR-AUC", "ROC-AUC", "Balanced accuracy"],
        key="cv_metric"
    )

    mean_column = f"{metric} mean"
    std_column = f"{metric} std"

    chart_data = cv_results.sort_values(mean_column, ascending=True)

    fig = px.bar(
        chart_data,
        x=mean_column,
        y="Model",
        orientation="h",
        error_x=std_column,
        text=mean_column
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=35, t=10, b=20),
        xaxis_title=metric,
        yaxis_title=None
    )

    st.plotly_chart(fig, width="stretch", height=450, config={"displayModeBar": False})

    best_cv = cv_results.loc[cv_results["PR-AUC mean"].idxmax()]

    col1, col2, col3 = st.columns(3)

    col1.metric("Best CV Model", best_cv["Model"])
    col2.metric("Mean PR-AUC", f"{best_cv['PR-AUC mean']:.3f}")
    col3.metric("PR-AUC Std", f"{best_cv['PR-AUC std']:.3f}")

    st.info(
        "The small cross-validation standard deviations indicate that performance "
        "was relatively stable across the five training folds."
    )

    with st.expander("View full cross-validation results"):
        display = cv_results.copy()
        numeric_columns = display.select_dtypes(include="number").columns
        display[numeric_columns] = display[numeric_columns].round(3)

        st.dataframe(display, width="stretch", hide_index=True)

with tab2:
    st.subheader("Validation Model Comparison")

    metric = st.selectbox(
        "Validation metric",
        [
            "PR-AUC",
            "ROC-AUC",
            "Diabetes precision",
            "Diabetes recall",
            "Diabetes F1",
            "Balanced accuracy",
            "Accuracy"
        ],
        key="validation_metric"
    )

    chart_data = validation_results.sort_values(metric, ascending=True)

    fig = px.bar(
        chart_data,
        x=metric,
        y="Model",
        orientation="h",
        text=metric
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=35, t=10, b=20),
        xaxis_title=metric,
        yaxis_title=None
    )

    st.plotly_chart(fig, width="stretch", height=450, config={"displayModeBar": False})

    st.info(
        "XGBoost achieved the strongest validation discrimination, although the "
        "improvement over Logistic Regression and Random Forest was relatively modest."
    )

    with st.expander("View full validation results"):
        display = validation_results.copy()
        numeric_columns = display.select_dtypes(include="number").columns
        display[numeric_columns] = display[numeric_columns].round(3)

        st.dataframe(display, width="stretch", hide_index=True)

with tab3:
    st.subheader("Effect of Class Weighting")

    st.write(
        """
        Class-weighted models place greater importance on correctly identifying
        diabetes cases. This primarily changes the precision-recall trade-off rather
        than substantially improving ranking performance.
        """
    )

    xgb_comparison = validation_results[
        validation_results["Model"].isin(["XGBoost", "XGBoost — weighted"])
    ][["Model", "Diabetes precision", "Diabetes recall", "Diabetes F1"]].copy()

    long_comparison = xgb_comparison.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    long_comparison["Metric"] = long_comparison["Metric"].str.replace("Diabetes ", "")

    fig = px.bar(
        long_comparison,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score"
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title=None,
        yaxis_title="Score",
        legend_title=None
    )

    st.plotly_chart(fig, width="stretch", height=380, config={"displayModeBar": False})

    standard = validation_results[validation_results["Model"] == "XGBoost"].iloc[0]
    weighted = validation_results[validation_results["Model"] == "XGBoost — weighted"].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recall",
        f"{weighted['Diabetes recall']:.1%}",
        f"{weighted['Diabetes recall'] - standard['Diabetes recall']:+.1%}"
    )

    col2.metric(
        "Precision",
        f"{weighted['Diabetes precision']:.1%}",
        f"{weighted['Diabetes precision'] - standard['Diabetes precision']:+.1%}"
    )

    col3.metric(
        "PR-AUC",
        f"{weighted['PR-AUC']:.3f}",
        f"{weighted['PR-AUC'] - standard['PR-AUC']:+.3f}"
    )

    st.warning(
        "Weighting substantially improves minority-class recall but produces many "
        "more false-positive predictions. ROC-AUC and PR-AUC remain almost unchanged."
    )

with tab4:
    st.subheader("Final Test Evaluation")

    st.write(
        """
        After model selection was completed using training cross-validation and
        validation PR-AUC and balanced accuracy, the selected Weighted XGBoost model was evaluated once on the
        untouched test set.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Diabetes Precision", f"{final_result['Diabetes precision']:.1%}")
    col2.metric("Diabetes Recall", f"{final_result['Diabetes recall']:.1%}")
    col3.metric("Balanced Accuracy", f"{final_result['Balanced accuracy']:.3f}")

    display = test_results.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)

    st.dataframe(display, width="stretch", hide_index=True)

    st.warning(
        "Class weighting substantially improves diabetes recall, but increases false-positive "
        "predictions. This model is intended for analytical demonstration only and should not "
        "be used for clinical screening or diagnosis."
    )

st.divider()

with st.expander("Selected Weighted XGBoost hyperparameters"):
    selected_model = best_params.get("selected_model", "XGBoost weighted")

    if selected_model in best_params:
        st.json(best_params[selected_model])
    else:
        st.write("No saved hyperparameters were found for the selected model.")

st.caption(
    "Final model selection was based on validation PR-AUC and balanced accuracy. The test set was used "
    "once after model selection."
)