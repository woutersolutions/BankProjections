import datetime
import os

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.valuation import Valuation
from bank_projections.scenarios.template_registry import TemplateRegistry
from bank_projections.utils.time import TimeHorizon
from examples import EXAMPLE_FOLDER
from examples.synthetic_data import create_synthetic_balance_sheet
from examples.ui.colors import get_chart_colors
from examples.ui.styles import apply_custom_styles


def main() -> None:
    st.set_page_config(page_title="Bank Projections - Example Model Run", layout="wide")
    apply_custom_styles()

    st.title("Bank Projections - Example Model Run")

    if "projection_result" not in st.session_state:
        st.session_state.projection_result = None

    config_path = os.path.join(EXAMPLE_FOLDER, "example_bs.csv")

    st.header("1. Balance Sheet Configuration")
    st.write("Edit the balance sheet configuration below:")

    if "config_df" not in st.session_state:
        st.session_state.config_df = pl.read_csv(config_path)

    config_df_pandas = st.session_state.config_df.to_pandas()
    edited_df = st.data_editor(config_df_pandas, use_container_width=True, height=400, num_rows="dynamic")
    st.session_state.config_df = pl.from_pandas(edited_df)

    st.divider()

    st.header("2. Time Horizon Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2024, 12, 31))

    with col2:
        number_of_months = st.number_input("Number of Months", value=24, min_value=1, max_value=600)

    with col3:
        end_of_month = st.checkbox("End of Month", value=True)

    st.divider()

    if st.button("Run Projection", type="primary", use_container_width=True):
        with st.spinner("Running projection... This may take a while."):
            scenario = TemplateRegistry.load_folder(os.path.join(EXAMPLE_FOLDER, "scenarios"))
            scenario.rules = {"Runoff": Runoff(), "Valuation": Valuation(), **scenario.rules}
            horizon = TimeHorizon.from_numbers(
                start_date=start_date,
                number_of_months=int(number_of_months),
                end_of_month=end_of_month,
            )

            start_bs = create_synthetic_balance_sheet(start_date, scenario, config_table=st.session_state.config_df)
            projection = Projection({"base": scenario}, horizon)
            result = projection.run(start_bs)
            st.session_state.projection_result = result

        st.success("Projection completed successfully!")

    if st.session_state.projection_result is not None:
        result = st.session_state.projection_result

        st.divider()
        st.header("3. Results")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Balance Sheet", "Income Statement", "Cashflow Statement", "Metrics", "Profitability"]
        )

        with tab1:
            st.subheader("Balance Sheet Book Values Over Time")
            group_by_bs = st.selectbox(
                "Group by",
                ["BalanceSheetSide", "ItemType", "SubItemType"],
                index=0,
                key="bs_group",
            )

            bs_df = pl.concat(result.balance_sheets, how="diagonal")
            bs_pandas = bs_df.to_pandas()

            if len(bs_pandas) > 0:
                fig = create_time_series_plot(bs_pandas, "ProjectionDate", "BookValue", group_by_bs, "Balance Sheet")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(bs_pandas, use_container_width=True)
            else:
                st.warning("No balance sheet data available")

        with tab2:
            st.subheader("Income Statement Over Time")
            group_by_pnl = st.selectbox(
                "Group by",
                ["ItemType", "SubItemType", "Rule"],
                index=0,
                key="pnl_group",
            )

            pnl_df = pl.concat(result.pnls, how="diagonal")
            pnl_pandas = pnl_df.to_pandas()

            if len(pnl_pandas) > 0:
                fig = create_time_series_plot(pnl_pandas, "ProjectionDate", "Amount", group_by_pnl, "Income Statement")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(pnl_pandas, use_container_width=True)
            else:
                st.warning("No P&L data available")

        with tab3:
            st.subheader("Cashflow Statement Over Time")
            group_by_cf = st.selectbox(
                "Group by",
                ["ItemType", "SubItemType", "Rule"],
                index=0,
                key="cf_group",
            )

            cf_df = pl.concat(result.cashflows, how="diagonal")
            cf_pandas = cf_df.to_pandas()

            if len(cf_pandas) > 0:
                fig = create_time_series_plot(cf_pandas, "ProjectionDate", "Amount", group_by_cf, "Cashflow Statement")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(cf_pandas, use_container_width=True)
            else:
                st.warning("No cashflow data available")

        with tab4:
            st.subheader("Metrics Over Time")

            metrics_df = pl.concat(result.metric_list, how="diagonal")
            metrics_pandas = metrics_df.to_pandas()

            if len(metrics_pandas) > 0:
                metric_columns = [col for col in metrics_pandas.columns if col not in ["Scenario", "ProjectionDate"]]
                selected_metrics = st.multiselect(
                    "Select metrics to plot",
                    metric_columns,
                    default=metric_columns[:3] if len(metric_columns) >= 3 else metric_columns,
                )

                if selected_metrics:
                    fig = create_metrics_plot(metrics_pandas, selected_metrics)
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(metrics_pandas, use_container_width=True)
            else:
                st.warning("No metrics data available")

        with tab5:
            st.subheader("Profitability Metrics Over Time")

            profitability_df = pl.concat(result.profitability_list, how="diagonal")
            profitability_pandas = profitability_df.to_pandas()

            if len(profitability_pandas) > 0:
                outlook_col = st.selectbox(
                    "Select outlook",
                    ["Monthly", "Quarterly", "Annual"],
                    index=1,
                    key="prof_outlook",
                )

                profitability_columns = [
                    col
                    for col in profitability_pandas.columns
                    if col not in ["Scenario", "ProjectionDate"] and outlook_col in col
                ]

                selected_prof_metrics = st.multiselect(
                    "Select profitability metrics to plot",
                    profitability_columns,
                    default=profitability_columns[:3] if len(profitability_columns) >= 3 else profitability_columns,
                )

                if selected_prof_metrics:
                    fig = create_metrics_plot(profitability_pandas, selected_prof_metrics)
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(profitability_pandas, use_container_width=True)
            else:
                st.warning("No profitability data available")


def create_time_series_plot(
    df: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str
) -> px.line:
    df_grouped = df.groupby([x_col, group_col])[y_col].sum().reset_index()

    fig = px.line(
        df_grouped,
        x=x_col,
        y=y_col,
        color=group_col,
        title=title,
        labels={x_col: "Date", y_col: "Value"},
        color_discrete_sequence=get_chart_colors(),
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title=group_col,
    )

    return fig


def create_metrics_plot(df: pd.DataFrame, metric_columns: list[str]) -> px.line:
    df_melted = df.melt(id_vars=["ProjectionDate"], value_vars=metric_columns, var_name="Metric", value_name="Value")

    fig = px.line(
        df_melted,
        x="ProjectionDate",
        y="Value",
        color="Metric",
        title="Selected Metrics Over Time",
        labels={"ProjectionDate": "Date", "Value": "Metric Value"},
        color_discrete_sequence=get_chart_colors(),
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Metric",
    )

    return fig


if __name__ == "__main__":
    main()
