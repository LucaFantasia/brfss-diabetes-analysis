"""
Main entry point for the CDC BRFSS Diabetes Analysis dashboard.

This file configures the Streamlit application and
defines navigation between the four dashboard pages.
"""

from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="BRFSS Diabetes Analysis",
    page_icon="📊",
    layout="wide"
)

# Keep the dashboard from becoming excessively wide on large displays.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.08);
            border: 1px solid rgba(120, 120, 120, 0.18);
            padding: 16px;
            border-radius: 10px;
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        h1 {
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

pages = [
    st.Page(APP_DIR / "pages" / "overview.py", title="Overview", icon="🏠", default=True),
    st.Page(APP_DIR / "pages" / "exploratory_analysis.py", title="Exploratory Data Analysis", icon="🔎"),
    st.Page(APP_DIR / "pages" / "statistical_analysis.py", title="Statistical Analysis", icon="📐"),
    st.Page(APP_DIR / "pages" / "model_performance.py", title="Model Performance", icon="📈")
]

page = st.navigation(pages)
page.run()