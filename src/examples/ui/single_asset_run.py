import datetime
import os

import plotly.express as px
import polars as pl
import streamlit as st

from bank_projections.output_config import AggregationConfig
from bank_projections.projections.projection import Projection
from bank_projections.projections.redemption import Redemption
from bank_projections.projections.valuation import Valuation
from bank_projections.scenarios.template_registry import TemplateRegistry
from bank_projections.utils.time import TimeHorizon
from examples import EXAMPLE_FOLDER
from examples.synthetic_data import create_single_asset_balance_sheet
from examples.ui.colors import get_chart_colors
from examples.ui.styles import apply_custom_styles

PLOTLY_CONFIG = {"width": "stretch"}


def main() -> None:
    st.set_page_config(page_title="Bank Projections - Single Asset Run", layout="wide")
    apply_custom_styles()

    st.title("Bank Projections - Single Asset Runoff")

    if "single_asset_result" not in st.session_state:
        st.session_state.single_asset_result = None

    st.header("1. Asset Configuration")

    col1, col2 = st.columns(2)

    with col1:
        book_value = st.number_input("Book Value", value=100000.0, min_value=0.0, step=1000.0)
        accounting_method = st.selectbox(
            "Accounting Method",
            ["amortized cost", "fair value through oci", "fair value through P&L"],
            index=0,
        )
        redemption_type = st.selectbox("Redemption Type", ["bullet", "linear", "annuity", "perpetual"], index=2)
        coupon_frequency = st.selectbox(
            "Coupon Frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Semi-Annual", "Annual", "Never"], index=2
        )
        coupon_type = st.selectbox("Coupon Type", ["fixed", "floating"], index=0)

    with col2:
        prepayment_rate = st.number_input("Prepayment Rate", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
        maturity = st.number_input("Maturity (years)", value=10, min_value=1, max_value=50, step=1)
        interest_rate = st.number_input("Interest Rate", value=0.03, min_value=0.0, max_value=1.0, step=0.001)
        valuation_method = st.selectbox(
            "Valuation Method",
            ["none", "amortizedcost", "fixedratebond", "floatingratebond", "swap"],
            index=0,
        )

    st.divider()

    st.header("2. Time Horizon Configuration")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2024, 12, 31))

    with col2:
        end_of_month = st.checkbox("End of Month", value=True, key="eom_sa")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        number_of_days = st.number_input("Number of Days", value=0, min_value=0, max_value=365, step=1, key="days_sa")

    with col2:
        number_of_weeks = st.number_input(
            "Number of Weeks", value=0, min_value=0, max_value=260, step=1, key="weeks_sa"
        )

    with col3:
        number_of_months = st.number_input(
            "Number of Months", value=24, min_value=0, max_value=600, step=1, key="months_sa"
        )

    with col4:
        number_of_quarters = st.number_input(
            "Number of Quarters", value=0, min_value=0, max_value=200, step=1, key="quarters_sa"
        )

    with col5:
        number_of_years = st.number_input("Number of Years", value=0, min_value=0, max_value=50, step=1, key="years_sa")

    st.divider()

    if st.button("Run Projection", type="primary", width="stretch"):
        # Load scenario but strip down to only basic rules needed for single asset runoff
        scenario = TemplateRegistry.load_folder(os.path.join(EXAMPLE_FOLDER, "scenarios"))
        # Keep only Runoff and Valuation - remove all other rules to avoid errors with missing items
        scenario.rules = {"Runoff": Redemption(), "Valuation": Valuation()}
        horizon = TimeHorizon.from_numbers(
            start_date=start_date,
            number_of_days=int(number_of_days),
            number_of_weeks=int(number_of_weeks),
            number_of_months=int(number_of_months),
            number_of_quarters=int(number_of_quarters),
            number_of_years=int(number_of_years),
            end_of_month=end_of_month,
        )

        start_bs = create_single_asset_balance_sheet(
            current_date=start_date,
            scenario=scenario,
            book_value=book_value,
            accounting_method=accounting_method,
            redemption_type=redemption_type,
            coupon_frequency=coupon_frequency,
            coupon_type=coupon_type,
            prepayment_rate=prepayment_rate,
            maturity=maturity,
            interest_rate=interest_rate,
            valuation_method=valuation_method if valuation_method != "none" else None,
        )
        projection = Projection({"base": scenario}, horizon)

        # Create progress bar
        progress_bar = st.progress(0, text="Starting projection...")
        status_text = st.empty()

        def update_progress(current: int, total: int) -> None:
            progress = current / total
            progress_bar.progress(progress, text=f"Running projection: {current}/{total} time steps")
            status_text.text(f"Progress: {current}/{total} ({progress * 100:.1f}%)")

        # Use AggregationConfig with all None to disable aggregation (show individual positions)
        no_aggregation = AggregationConfig()
        result = projection.run(start_bs, no_aggregation, progress_callback=update_progress)
        st.session_state.single_asset_result = result

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        st.success("Projection completed successfully!")

    if st.session_state.single_asset_result is not None:
        result = st.session_state.single_asset_result

        st.divider()
        st.header("3. Results")

        tab1, tab2, tab3 = st.tabs(["Asset Nominal", "P&L by Rule", "Cashflows by Rule"])

        with tab1:
            st.subheader("Asset Nominal Over Time")

            bs_df = pl.concat(result.balance_sheets, how="diagonal")
            bs_pandas = bs_df.to_pandas()

            # Filter to show only the loan asset
            if len(bs_pandas) > 0:
                bs_filtered = bs_pandas[bs_pandas["ItemType"] == "Loan"]

                if len(bs_filtered) > 0:
                    fig = px.line(
                        bs_filtered,
                        x="ProjectionDate",
                        y="Nominal",
                        title="Asset Nominal Over Time",
                        labels={"ProjectionDate": "Date", "Nominal": "Nominal"},
                        color_discrete_sequence=get_chart_colors(),
                    )

                    fig.update_layout(
                        hovermode="x unified",
                        xaxis_title="Date",
                        yaxis_title="Nominal",
                    )

                    st.plotly_chart(fig, config=PLOTLY_CONFIG)

                    with st.expander("View Raw Data"):
                        st.dataframe(bs_filtered, width="stretch")
                else:
                    st.warning("No loan asset data available")
            else:
                st.warning("No balance sheet data available")

        with tab2:
            st.subheader("P&L by Rule Over Time")

            pnl_df = pl.concat(result.pnls, how="diagonal")
            pnl_pandas = pnl_df.to_pandas()

            # Filter to show only the loan asset P&L
            if len(pnl_pandas) > 0:
                pnl_filtered = pnl_pandas[pnl_pandas["ItemType"] == "Loan"]
                pnl_grouped = pnl_filtered.groupby(["ProjectionDate", "rule"])["Amount"].sum().reset_index()

                fig = px.bar(
                    pnl_grouped,
                    x="ProjectionDate",
                    y="Amount",
                    color="rule",
                    title="P&L by Rule Over Time",
                    labels={"ProjectionDate": "Date", "Amount": "Amount"},
                    color_discrete_sequence=get_chart_colors(),
                )

                fig.update_layout(
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Amount",
                    legend_title="Rule",
                )

                st.plotly_chart(fig, config=PLOTLY_CONFIG)

                with st.expander("View Raw Data"):
                    st.dataframe(pnl_filtered, width="stretch")
            else:
                st.warning("No P&L data available")

        with tab3:
            st.subheader("Cashflows by Rule Over Time")

            cf_df = pl.concat(result.cashflows, how="diagonal")
            cf_pandas = cf_df.to_pandas()

            # Filter to show only the loan asset cashflows
            if len(cf_pandas) > 0:
                cf_filtered = cf_pandas[cf_pandas["ItemType"] == "Loan"]
                cf_grouped = cf_filtered.groupby(["ProjectionDate", "rule"])["Amount"].sum().reset_index()

                fig = px.bar(
                    cf_grouped,
                    x="ProjectionDate",
                    y="Amount",
                    color="rule",
                    title="Cashflows by Rule Over Time",
                    labels={"ProjectionDate": "Date", "Amount": "Amount"},
                    color_discrete_sequence=get_chart_colors(),
                )

                fig.update_layout(
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Amount",
                    legend_title="Rule",
                )

                st.plotly_chart(fig, config=PLOTLY_CONFIG)

                with st.expander("View Raw Data"):
                    st.dataframe(cf_filtered, width="stretch")
            else:
                st.warning("No cashflow data available")


if __name__ == "__main__":
    main()
