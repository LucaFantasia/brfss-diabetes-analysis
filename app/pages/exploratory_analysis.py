"""
Exploratory Analysis page for the BRFSS Diabetes Analysis dashboard.

This page allows users to interactively explore diabetes prevalence across
ordinal, binary and quantitative health indicators.
"""

import plotly.express as px
import streamlit as st

from utils import (
    EDA_INSIGHTS,
    FEATURE_LABELS,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    QUANTITATIVE_FEATURES,
    binary_prevalence,
    load_data,
    ordinal_prevalence,
    quantitative_prevalence
)


df = load_data()

st.title("Exploratory Analysis")
st.caption("Explore how diabetes prevalence varies across the BRFSS health indicators.")

st.write(
    """
    Select a feature group and variable to explore its relationship with
    self-reported diabetes status. These are descriptive associations and should
    not be interpreted as causal effects.
    """
)

feature_type = st.radio(
    "Feature group",
    ["Ordinal", "Binary", "Quantitative"],
    horizontal=True
)

if feature_type == "Ordinal":
    feature = st.selectbox(
        "Select a feature",
        ORDINAL_FEATURES,
        format_func=lambda value: FEATURE_LABELS[value]
    )

    summary = ordinal_prevalence(df, feature)

    chart_col, insight_col = st.columns([1.8, 1])

    with chart_col:
        fig = px.line(
            summary,
            x="Category",
            y="Diabetes prevalence",
            markers=True,
            custom_data=["size"]
        )

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Diabetes prevalence: %{y:.1f}%"
            "<br>Respondents: %{customdata[0]:,}<extra></extra>"
        )

        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title=None,
            yaxis_title="Diabetes prevalence (%)"
        )

        st.plotly_chart(fig, width="stretch", height=380, config={"displayModeBar": False})

    with insight_col:
        st.subheader("Key Observation")
        st.write(EDA_INSIGHTS[feature])

        min_prevalence = summary["Diabetes prevalence"].min()
        max_prevalence = summary["Diabetes prevalence"].max()

        st.metric("Lowest Prevalence", f"{min_prevalence:.1f}%")
        st.metric("Highest Prevalence", f"{max_prevalence:.1f}%")
        st.metric("Difference", f"{max_prevalence - min_prevalence:.1f} pp")

elif feature_type == "Binary":
    feature = st.selectbox(
        "Select a feature",
        NOMINAL_FEATURES,
        format_func=lambda value: FEATURE_LABELS[value]
    )

    summary = binary_prevalence(df, feature)

    chart_col, insight_col = st.columns([1.8, 1])

    with chart_col:
        fig = px.bar(
            summary,
            x="Category",
            y="Diabetes prevalence",
            text="Diabetes prevalence",
            custom_data=["size"]
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Diabetes prevalence: %{y:.1f}%"
            "<br>Respondents: %{customdata[0]:,}<extra></extra>"
        )

        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title=None,
            yaxis_title="Diabetes prevalence (%)",
            showlegend=False
        )

        st.plotly_chart(fig, width="stretch", height=360, config={"displayModeBar": False})

    with insight_col:
        prevalence_0 = summary.iloc[0]["Diabetes prevalence"]
        prevalence_1 = summary.iloc[1]["Diabetes prevalence"]
        difference = prevalence_1 - prevalence_0

        st.subheader("Prevalence Comparison")
        st.metric(summary.iloc[0]["Category"], f"{prevalence_0:.1f}%")
        st.metric(summary.iloc[1]["Category"], f"{prevalence_1:.1f}%")
        st.metric("Difference", f"{difference:+.1f} pp")

        st.caption(
            "A positive difference means diabetes prevalence is higher for the "
            "second category. This does not imply causation."
        )

else:
    feature = st.selectbox(
        "Select a feature",
        QUANTITATIVE_FEATURES,
        format_func=lambda value: FEATURE_LABELS[value]
    )

    summary = quantitative_prevalence(df, feature)

    chart_col, insight_col = st.columns([1.8, 1])

    with chart_col:
        fig = px.bar(
            summary,
            x="Category",
            y="Diabetes prevalence",
            text="Diabetes prevalence",
            custom_data=["size"]
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Diabetes prevalence: %{y:.1f}%"
            "<br>Respondents: %{customdata[0]:,}<extra></extra>"
        )

        fig.update_layout(
            height=370,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title=FEATURE_LABELS[feature],
            yaxis_title="Diabetes prevalence (%)",
            showlegend=False
        )

        st.plotly_chart(fig, width="stretch", height=370, config={"displayModeBar": False})

    with insight_col:
        st.subheader("Key Observation")
        st.write(EDA_INSIGHTS[feature])

        min_prevalence = summary["Diabetes prevalence"].min()
        max_prevalence = summary["Diabetes prevalence"].max()

        st.metric("Lowest Prevalence", f"{min_prevalence:.1f}%")
        st.metric("Highest Prevalence", f"{max_prevalence:.1f}%")
        st.metric("Range", f"{max_prevalence - min_prevalence:.1f} pp")

st.divider()
st.subheader("Key EDA Findings")

st.markdown(
    """
    - Diabetes prevalence rises substantially as **self-reported general health worsens**.
    - Prevalence is generally higher among **older respondents**.
    - **High blood pressure, high cholesterol, difficulty walking and cardiovascular history**
      show notable prevalence differences.
    - **BMI** shows a particularly clear relationship with diabetes prevalence.
    - Poor physical health shows a stronger pattern than poor mental-health days.
    """
)