import datetime
import os

import plotly.express as px
import polars as pl
import streamlit as st

from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.valuation import Valuation
from bank_projections.scenarios.template_registry import TemplateRegistry
from bank_projections.utils.time import TimeHorizon
from examples import EXAMPLE_FOLDER
from examples.synthetic_data import create_single_asset_balance_sheet
from examples.ui.colors import get_chart_colors
from examples.ui.styles import apply_custom_styles


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
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2024, 12, 31))

    with col2:
        number_of_months = st.number_input("Number of Months", value=24, min_value=1, max_value=600, key="months_sa")

    with col3:
        end_of_month = st.checkbox("End of Month", value=True, key="eom_sa")

    st.divider()

    if st.button("Run Projection", type="primary", use_container_width=True):
        with st.spinner("Running projection..."):
            scenario = TemplateRegistry.load_folder(os.path.join(EXAMPLE_FOLDER, "scenarios"))
            scenario.rules = {"Runoff": Runoff(), "Valuation": Valuation(), **scenario.rules}
            horizon = TimeHorizon.from_numbers(
                start_date=start_date,
                number_of_months=int(number_of_months),
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
            result = projection.run(start_bs)
            st.session_state.single_asset_result = result

        st.success("Projection completed successfully!")

    if st.session_state.single_asset_result is not None:
        result = st.session_state.single_asset_result

        st.divider()
        st.header("3. Results")

        tab1, tab2, tab3 = st.tabs(["Asset Quantity", "P&L by Rule", "Cashflows by Rule"])

        with tab1:
            st.subheader("Asset Quantity Over Time")

            bs_df = pl.concat(result.balance_sheets, how="diagonal")
            bs_pandas = bs_df.to_pandas()

            if len(bs_pandas) > 0:
                fig = px.line(
                    bs_pandas,
                    x="ProjectionDate",
                    y="Quantity",
                    title="Asset Quantity Over Time",
                    labels={"ProjectionDate": "Date", "Quantity": "Quantity"},
                    color_discrete_sequence=get_chart_colors(),
                )

                fig.update_layout(
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Quantity",
                )

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(bs_pandas, use_container_width=True)
            else:
                st.warning("No balance sheet data available")

        with tab2:
            st.subheader("P&L by Rule Over Time")

            pnl_df = pl.concat(result.pnls, how="diagonal")
            pnl_pandas = pnl_df.to_pandas()

            if len(pnl_pandas) > 0:
                pnl_grouped = pnl_pandas.groupby(["ProjectionDate", "Rule"])["Amount"].sum().reset_index()

                fig = px.bar(
                    pnl_grouped,
                    x="ProjectionDate",
                    y="Amount",
                    color="Rule",
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

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(pnl_pandas, use_container_width=True)
            else:
                st.warning("No P&L data available")

        with tab3:
            st.subheader("Cashflows by Rule Over Time")

            cf_df = pl.concat(result.cashflows, how="diagonal")
            cf_pandas = cf_df.to_pandas()

            if len(cf_pandas) > 0:
                cf_grouped = cf_pandas.groupby(["ProjectionDate", "Rule"])["Amount"].sum().reset_index()

                fig = px.bar(
                    cf_grouped,
                    x="ProjectionDate",
                    y="Amount",
                    color="Rule",
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

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(cf_pandas, use_container_width=True)
            else:
                st.warning("No cashflow data available")


if __name__ == "__main__":
    main()
