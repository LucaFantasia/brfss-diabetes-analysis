"""
Statistical Analysis page for the BRFSS Diabetes Analysis dashboard.

This page presents statistical association tests and effect sizes for binary,
ordinal and quantitative features.
"""

import plotly.express as px
import streamlit as st

from utils import (
    calculate_statistical_results, format_p_value, load_data
)


df = load_data()

binary_results, ordinal_results, quantitative_results = calculate_statistical_results(df)

st.title("Statistical Analysis")
st.caption("Quantifying the relationships observed during exploratory analysis.")

st.write(
    """
    Different statistical tests are used according to feature type. Because the
    dataset contains more than 250,000 observations, very small p-values are common.
    Effect sizes are therefore more useful than statistical significance alone.
    """
)

strongest_binary = binary_results.iloc[0]
strongest_ordinal = ordinal_results.iloc[0]
strongest_quantitative = quantitative_results.iloc[0]

col1, col2, col3 = st.columns(3)

binary_effect = strongest_binary["Cramér's V"]
col1.metric(
    "Strongest Binary Association",
    strongest_binary["Feature"],
    f"Cramér's V = {binary_effect:.3f}"
)

col2.metric(
    "Strongest Ordinal Association",
    strongest_ordinal["Feature"],
    f"ρ = {strongest_ordinal['Spearman rho']:+.3f}"
)

col3.metric(
    "Strongest Quantitative Effect",
    strongest_quantitative["Feature"],
    f"r = {strongest_quantitative['Rank-biserial']:+.3f}"
)

st.divider()

tab1, tab2, tab3 = st.tabs(["Binary Features", "Ordinal Features", "Quantitative Features"])

with tab1:
    st.subheader("Binary Feature Associations")

    st.write(
        """
        Chi-square tests assess whether each binary indicator is associated with
        diabetes status. **Cramér's V** measures the strength of the association.
        """
    )

    chart_data = binary_results.sort_values("Cramér's V", ascending=True)

    fig = px.bar(
        chart_data,
        x="Cramér's V",
        y="Feature",
        orientation="h",
        text="Cramér's V"
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

    fig.update_layout(
        height=470,
        margin=dict(l=20, r=30, t=10, b=20),
        xaxis_title="Cramér's V",
        yaxis_title=None
    )

    st.plotly_chart(fig, width="stretch", height=470, config={"displayModeBar": False})

    st.info(
        "High blood pressure, difficulty walking, high cholesterol and cardiovascular "
        "history show some of the strongest binary associations with diabetes."
    )

    with st.expander("View statistical results table"):
        display = binary_results.copy()
        display["Cramér's V"] = display["Cramér's V"].round(3)
        display["p-value"] = display["p-value"].map(format_p_value)
        display["Adjusted p-value"] = display["Adjusted p-value"].map(format_p_value)

        st.dataframe(display, width="stretch", hide_index=True)

with tab2:
    st.subheader("Ordinal Feature Associations")

    st.write(
        """
        **Spearman rank correlation** measures the direction and strength of the
        relationship between each ordered feature and diabetes status.
        """
    )

    chart_data = ordinal_results.sort_values("Spearman rho", ascending=True)

    fig = px.bar(
        chart_data,
        x="Spearman rho",
        y="Feature",
        orientation="h",
        text="Spearman rho"
    )

    fig.update_traces(texttemplate="%{text:+.3f}", textposition="outside")

    fig.update_layout(
        height=310,
        margin=dict(l=20, r=30, t=10, b=20),
        xaxis_title="Spearman rho",
        yaxis_title=None
    )

    st.plotly_chart(fig, width="stretch", height=310, config={"displayModeBar": False})

    st.info(
        "General health shows the strongest ordinal association. Age is positively "
        "associated with diabetes, while income and education show negative associations."
    )

    with st.expander("View statistical results table"):
        display = ordinal_results.copy()
        display["Spearman rho"] = display["Spearman rho"].round(3)
        display["p-value"] = display["p-value"].map(format_p_value)
        display["Adjusted p-value"] = display["Adjusted p-value"].map(format_p_value)

        st.dataframe(display, width="stretch", hide_index=True)

with tab3:
    st.subheader("Quantitative Feature Associations")

    st.write(
        """
        **Mann-Whitney U tests** compare the distributions of each quantitative
        variable between respondents with and without diabetes. Rank-biserial
        correlation is reported as the effect size.
        """
    )

    chart_data = quantitative_results.sort_values("Rank-biserial", ascending=True)

    fig = px.bar(
        chart_data,
        x="Rank-biserial",
        y="Feature",
        orientation="h",
        text="Rank-biserial"
    )

    fig.update_traces(texttemplate="%{text:+.3f}", textposition="outside")

    fig.update_layout(
        height=290,
        margin=dict(l=20, r=30, t=10, b=20),
        xaxis_title="Rank-biserial correlation",
        yaxis_title=None
    )

    st.plotly_chart(fig, width="stretch", height=290, config={"displayModeBar": False})

    st.info(
        "BMI shows the clearest quantitative separation, followed by poor physical "
        "health. Poor mental-health days show a considerably weaker effect."
    )

    with st.expander("View statistical results table"):
        display = quantitative_results.copy()

        display["Median: no diabetes"] = display["Median: no diabetes"].round(1)
        display["Median: diabetes"] = display["Median: diabetes"].round(1)
        display["Rank-biserial"] = display["Rank-biserial"].round(3)
        display["p-value"] = display["p-value"].map(format_p_value)
        display["Adjusted p-value"] = display["Adjusted p-value"].map(format_p_value)

        st.dataframe(display, width="stretch", hide_index=True)

st.divider()

st.warning(
    "These tests examine individual associations. They do not establish causal "
    "relationships and do not account for interactions or confounding between predictors."
)