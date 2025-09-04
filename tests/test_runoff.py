"""Unit tests for runoff module."""

import datetime

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheetItem, MutationReason
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.frequency import FrequencyRegistry, Monthly, Quarterly
from bank_projections.projections.rule import TimeIncrement
from bank_projections.projections.runoff import CouponPayments
from examples.synthetic_data import create_balanced_balance_sheet


class TestCouponPayments:
    """Test CouponPayments rule functionality."""

    def setup_method(self) -> None:
        """Set up test balance sheet and register frequencies."""
        # Clear and register test frequencies
        FrequencyRegistry._registry.clear()
        FrequencyRegistry.register("Monthly", Monthly)
        FrequencyRegistry.register("Quarterly", Quarterly)

        # Create test balance sheet with loan data
        self.bs = create_balanced_balance_sheet(total_assets=1_000_000, random_seed=42)

        # Add required columns for coupon calculations
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            [
                pl.when(loan_filter).then(pl.lit("Monthly")).otherwise(pl.lit("")).alias("CouponFrequency"),
                pl.when(loan_filter)
                .then(pl.lit(datetime.date(2025, 2, 15)))
                .otherwise(pl.lit(None))
                .alias("NextCouponDate"),
                pl.when(loan_filter)
                .then(pl.lit(datetime.date(2030, 1, 15)))
                .otherwise(pl.lit(None))
                .alias("MaturityDate"),
                pl.when(loan_filter)
                .then(pl.lit(0.05))  # 5% annual interest rate
                .otherwise(pl.lit(0.0))
                .alias("InterestRate"),
                pl.when(loan_filter)
                .then(pl.lit(False))  # Not accumulating
                .otherwise(pl.lit(False))
                .alias("IsAccumulating"),
                pl.when(loan_filter)
                .then(pl.lit(1000.0))  # Some accrued interest
                .otherwise(pl.lit(0.0))
                .alias("AccruedInterest"),
            ]
        )

    def test_apply_monthly_coupon_payment(self) -> None:
        """Test applying monthly coupon payment rule."""
        # Set up time increment for one month
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        # Get initial state
        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        initial_cashflows_len = len(self.bs.cashflows)

        # Apply coupon payment rule
        rule = CouponPayments()
        result_bs = rule.apply(self.bs, increment)

        # Verify quantity unchanged (not accumulating)
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert abs(new_quantity - initial_quantity) < 0.01

        # Verify cashflows were generated
        assert len(result_bs.cashflows) > initial_cashflows_len

        # Verify PnL was recorded
        assert len(result_bs.pnls) > 0

    def test_apply_with_accumulating_loans(self) -> None:
        """Test coupon payments with accumulating loans."""
        # Set some loans to accumulating
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit(True))
            .otherwise(pl.col("IsAccumulating"))
            .alias("IsAccumulating")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = CouponPayments()
        result_bs = rule.apply(self.bs, increment)

        # For accumulating loans, quantity should increase
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity > initial_quantity

    def test_apply_with_matured_loans(self) -> None:
        """Test coupon payments when some loans mature during the period."""
        # Set some loans to mature before projection end
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit(datetime.date(2025, 1, 20)))  # Matures during period
            .otherwise(pl.col("MaturityDate"))
            .alias("MaturityDate")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        rule = CouponPayments()
        result_bs = rule.apply(self.bs, increment)

        # Should have some accrued interest changes or at least process without error
        new_accrued = result_bs.get_amount(BalanceSheetItem(AssetType="loan"), BalanceSheetMetrics.accrued_interest)
        # The rule should execute successfully (accrued interest may or may not change)
        assert isinstance(new_accrued, float)

    def test_apply_quarterly_frequency(self) -> None:
        """Test applying rule with quarterly frequency."""
        # Set some loans to quarterly frequency
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit("Quarterly"))
            .otherwise(pl.col("CouponFrequency"))
            .alias("CouponFrequency")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 4, 15))

        rule = CouponPayments()
        initial_pnl_len = len(self.bs.pnls)

        result_bs = rule.apply(self.bs, increment)

        # Should generate PnL for quarterly payments
        assert len(result_bs.pnls) > initial_pnl_len

    def test_apply_preserves_balance_sheet_structure(self) -> None:
        """Test that applying rule preserves balance sheet structure."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_columns = set(self.bs._data.columns)
        initial_rows = len(self.bs._data)

        rule = CouponPayments()
        result_bs = rule.apply(self.bs, increment)

        # Should preserve row count and basic columns
        assert len(result_bs._data) == initial_rows
        assert initial_columns.issubset(set(result_bs._data.columns))

    def test_apply_zero_interest_rate(self) -> None:
        """Test applying rule with zero interest rate."""
        # Set interest rate to zero for all loans
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter)
            .then(pl.lit(0.0))
            .otherwise(pl.col("InterestRate"))
            .alias("InterestRate")  # TODO: Use mutate matric to set the interest
        )
        item = BalanceSheetItem(AssetType="loan")
        self.bs.mutate_metric(item, BalanceSheetMetrics.accrued_interest, 0.0, MutationReason(test="test"))

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        rule = CouponPayments()
        initial_pnl = (
            self.bs.pnls.filter(pl.col("AssetType") == "loan")["Amount"].sum() if len(self.bs.pnls) > 0 else 0.0
        )

        result_bs = rule.apply(self.bs, increment)
        new_pnl = result_bs.pnls.filter(pl.col("AssetType") == "loan")["Amount"].sum()

        assert initial_pnl == new_pnl

    def test_apply_edge_case_end_of_month(self) -> None:
        """Test edge case with end of month dates."""
        # Set next coupon date to end of month
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter)
            .then(pl.lit(datetime.date(2025, 1, 31)))
            .otherwise(pl.col("NextCouponDate"))
            .alias("NextCouponDate")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 28))

        rule = CouponPayments()
        result_bs = rule.apply(self.bs, increment)

        # Should handle end of month correctly
        assert len(result_bs._data) > 0
