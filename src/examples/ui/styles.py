import streamlit as st

from examples.ui.colors import NEUTRAL_COLORS, PRIMARY_COLORS, SEMANTIC_COLORS


def apply_custom_styles() -> None:
    st.markdown(
        f"""
        <style>
        /* Headers and titles */
        h1, h2, h3 {{
            color: {PRIMARY_COLORS['deep_navy']} !important;
        }}

        /* Primary buttons */
        .stButton > button {{
            background-color: {PRIMARY_COLORS['bright_blue']} !important;
            color: {NEUTRAL_COLORS['background_white']} !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }}

        .stButton > button:hover {{
            background-color: {PRIMARY_COLORS['medium_blue']} !important;
        }}

        /* Success buttons */
        .stButton > button[kind="primary"] {{
            background-color: {SEMANTIC_COLORS['success_green']} !important;
        }}

        /* Text colors */
        .stMarkdown p {{
            color: {NEUTRAL_COLORS['medium_gray']} !important;
        }}

        /* Data editor and dataframe */
        .stDataFrame {{
            border: 1px solid {NEUTRAL_COLORS['border_gray']} !important;
        }}

        /* Metric cards */
        [data-testid="stMetricValue"] {{
            color: {PRIMARY_COLORS['deep_navy']} !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {NEUTRAL_COLORS['background_light']} !important;
        }}

        /* Dividers */
        hr {{
            border-color: {NEUTRAL_COLORS['border_gray']} !important;
        }}

        /* Info boxes */
        .stAlert {{
            background-color: {NEUTRAL_COLORS['surface_gray']} !important;
            border-left: 4px solid {SEMANTIC_COLORS['info_purple']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
