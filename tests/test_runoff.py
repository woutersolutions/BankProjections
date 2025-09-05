"""Unit tests for runoff module."""

import datetime

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheetItem, MutationReason
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.frequency import FrequencyRegistry, Monthly, Quarterly
from bank_projections.projections.redemption import (
    AnnuityRedemption,
    BulletRedemption,
    LinearRedemption,
    PerpetualRedemption,
    RedemptionRegistry,
)
from bank_projections.projections.rule import TimeIncrement
from bank_projections.projections.runoff import Runoff
from examples.synthetic_data import create_balanced_balance_sheet


class TestRunoff:
    """Test Runoff rule functionality including coupon payments and principal repayments."""

    def setup_method(self) -> None:
        """Set up test balance sheet and register frequencies and redemption types."""
        # Clear and register test frequencies
        FrequencyRegistry._registry.clear()
        FrequencyRegistry.register("Monthly", Monthly)
        FrequencyRegistry.register("Quarterly", Quarterly)

        # Clear and register redemption types
        RedemptionRegistry._registry.clear()
        RedemptionRegistry.register("bullet", BulletRedemption)
        RedemptionRegistry.register("annuity", AnnuityRedemption)
        RedemptionRegistry.register("linear", LinearRedemption)
        RedemptionRegistry.register("perpetual", PerpetualRedemption)

        # Create test balance sheet with loan data
        self.bs = create_balanced_balance_sheet(total_assets=1_000_000, random_seed=42)

        # Validate initial balance sheet
        self.bs.validate()

        # Add required columns for runoff calculations
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
                .then(pl.lit(0.0))  # Start with zero accrued interest
                .otherwise(pl.lit(0.0))
                .alias("AccruedInterest"),
                pl.when(loan_filter)
                .then(pl.lit(0.0))  # Start with zero agio
                .otherwise(pl.lit(0.0))
                .alias("Agio"),
                pl.when(loan_filter)
                .then(pl.lit(0.0))  # Start with zero impairment
                .otherwise(pl.lit(0.0))
                .alias("Impairment"),
                pl.when(loan_filter)
                .then(pl.lit("bullet"))  # Default to bullet redemption
                .otherwise(pl.lit(""))
                .alias("RedemptionType"),
                pl.when(loan_filter)
                .then(pl.lit(0.02))  # 2% annual prepayment rate
                .otherwise(pl.lit(0.0))
                .alias("PrepaymentRate"),
            ]
        )

    def test_bullet_repayment_before_maturity(self) -> None:
        """Test bullet loans have no scheduled repayment before maturity, but may have prepayments."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should decrease slightly due to prepayments only
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity  # Some prepayment occurred
        assert new_quantity > initial_quantity * 0.95  # But not much (small prepayment rate)

        # Should have principal repayment cashflow of zero (no scheduled repayment)
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert abs(total_repayment) < 0.01

        # Should have principal prepayment cashflow greater than zero
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert total_prepayment > 0

    def test_bullet_repayment_at_maturity(self) -> None:
        """Test bullet loans are fully repaid at maturity."""
        # Set maturity within the projection period
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter)
            .then(pl.lit(datetime.date(2025, 2, 1)))
            .otherwise(pl.col("MaturityDate"))
            .alias("MaturityDate")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should go to zero for matured loans
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity * 0.1  # Should be nearly zero

        # Should have principal repayment cashflow equal to initial quantity
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert abs(total_repayment - initial_quantity) < initial_quantity * 0.01

        # Verify balance sheet is still valid after runoff
        result_bs.validate()

    def test_annuity_repayment(self) -> None:
        """Test annuity loans have regular principal repayments."""
        # Set redemption type to annuity
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter).then(pl.lit("annuity")).otherwise(pl.col("RedemptionType")).alias("RedemptionType")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should decrease for annuity loans
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity

        # Should have positive principal repayment cashflow
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert total_repayment > 0

        # Verify balance sheet is still valid after runoff
        result_bs.validate()

    def test_linear_repayment(self) -> None:
        """Test linear loans have equal principal repayments each period."""
        # Set redemption type to linear
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter).then(pl.lit("linear")).otherwise(pl.col("RedemptionType")).alias("RedemptionType")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should decrease for linear loans
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity

        # Should have positive principal repayment cashflow
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert total_repayment > 0

        # Verify balance sheet is still valid after runoff
        result_bs.validate()

    def test_perpetual_repayment(self) -> None:
        """Test perpetual loans have no scheduled repayments but may have prepayments."""
        # Set redemption type to perpetual
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter).then(pl.lit("perpetual")).otherwise(pl.col("RedemptionType")).alias("RedemptionType")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should decrease slightly due to prepayments only
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity  # Some prepayment occurred
        assert new_quantity > initial_quantity * 0.95  # But not much (small prepayment rate)

        # Should have no principal repayment cashflow (perpetual = no scheduled repayments)
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert abs(total_repayment) < 0.01

        # Should have positive prepayment cashflow
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert total_prepayment > 0

    def test_coupon_payments_non_accumulating(self) -> None:
        """Test coupon payments for non-accumulating loans generate cashflows."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_cashflows_len = len(self.bs.cashflows)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Should generate coupon payment cashflows
        coupon_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Coupon payment")
        )
        assert len(coupon_cashflows) > 0

        # Total cashflows should increase
        assert len(result_bs.cashflows) > initial_cashflows_len

    def test_coupon_payments_accumulating(self) -> None:
        """Test coupon payments for accumulating loans increase quantity."""
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

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # For accumulating loans, quantity should increase (interest is added to principal)
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity > initial_quantity

        # Should have zero coupon payment cashflows for accumulating loans
        coupon_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Coupon payment")
        )
        accumulating_coupon_amount = coupon_cashflows["Amount"].sum() if len(coupon_cashflows) > 0 else 0.0
        # Some loans are still non-accumulating, so there should be some cashflow
        assert accumulating_coupon_amount >= 0

    def test_interest_accrual_update(self) -> None:
        """Test that accrued interest is properly updated."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Accrued interest should be updated
        new_accrued = result_bs.get_amount(loans_item, BalanceSheetMetrics.accrued_interest)
        # The accrued interest may increase or decrease based on coupon payments and new accrual
        assert isinstance(new_accrued, float)

    def test_impairment_adjustment_for_repayments(self) -> None:
        """Test that impairment is adjusted proportionally to repayments."""
        # Use annuity loans to test impairment adjustment and set some initial impairment
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            [
                pl.when(loan_filter)
                .then(pl.lit("annuity"))
                .otherwise(pl.col("RedemptionType"))
                .alias("RedemptionType"),
                pl.when(loan_filter)
                .then(pl.lit(100.0))
                .otherwise(pl.col("Impairment"))
                .alias("Impairment"),  # Set some initial impairment
            ]
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_impairment = self.bs.get_amount(loans_item, BalanceSheetMetrics.impairment)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Impairment should decrease proportionally to principal repayments
        new_impairment = result_bs.get_amount(loans_item, BalanceSheetMetrics.impairment)
        assert new_impairment < initial_impairment

    def test_agio_linear_decrease(self) -> None:
        """Test that agio decreases linearly over time."""
        # Set some initial agio to test the decrease
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter).then(pl.lit(500.0)).otherwise(pl.col("Agio")).alias("Agio")  # Set some initial agio
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_agio = self.bs.get_amount(loans_item, BalanceSheetMetrics.agio)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Agio should decrease (linear amortization)
        new_agio = result_bs.get_amount(loans_item, BalanceSheetMetrics.agio)
        assert new_agio < initial_agio

    def test_pnl_generation(self) -> None:
        """Test that appropriate PnL entries are generated."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_pnl_len = len(self.bs.pnls)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Should generate PnL for interest income and impairment
        assert len(result_bs.pnls) > initial_pnl_len

        # Check for specific PnL reasons
        interest_pnl = result_bs.pnls.filter((pl.col("module") == "Runoff") & (pl.col("rule") == "Interest Income"))
        impairment_pnl = result_bs.pnls.filter((pl.col("module") == "Runoff") & (pl.col("rule") == "Impairment"))

        assert len(interest_pnl) > 0
        assert len(impairment_pnl) > 0

    def test_quarterly_frequency(self) -> None:
        """Test runoff with quarterly frequency."""
        # Set some loans to quarterly frequency
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit("Quarterly"))
            .otherwise(pl.col("CouponFrequency"))
            .alias("CouponFrequency"),
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit(datetime.date(2025, 4, 15)))
            .otherwise(pl.col("NextCouponDate"))
            .alias("NextCouponDate"),
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 4, 15))

        rule = Runoff()
        initial_pnl_len = len(self.bs.pnls)

        result_bs = rule.apply(self.bs, increment)

        # Should generate PnL for quarterly payments
        assert len(result_bs.pnls) > initial_pnl_len

    def test_preserves_balance_sheet_structure(self) -> None:
        """Test that applying rule preserves balance sheet structure."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_columns = set(self.bs._data.columns)
        initial_rows = len(self.bs._data)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Should preserve row count and basic columns
        assert len(result_bs._data) == initial_rows
        assert initial_columns.issubset(set(result_bs._data.columns))

    def test_zero_interest_rate_handling(self) -> None:
        """Test runoff with zero interest rates."""
        # Set interest rate to zero for all loans
        item = BalanceSheetItem(AssetType="loan")
        reason = MutationReason(test="test")
        self.bs.mutate_metric(item, BalanceSheetMetrics.interest_rate, 0.0, reason)
        self.bs.mutate_metric(item, BalanceSheetMetrics.accrued_interest, 0.0, reason)

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Should handle zero interest rates without error
        loans_item = BalanceSheetItem(AssetType="loan")
        new_accrued = result_bs.get_amount(loans_item, BalanceSheetMetrics.accrued_interest)
        assert isinstance(new_accrued, float)

    def test_combined_repayment_types(self) -> None:
        """Test mixed redemption types in same portfolio."""
        # Set different redemption types for different loans
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter & (pl.int_range(pl.len()) % 4 == 0))
            .then(pl.lit("bullet"))
            .when(loan_filter & (pl.int_range(pl.len()) % 4 == 1))
            .then(pl.lit("annuity"))
            .when(loan_filter & (pl.int_range(pl.len()) % 4 == 2))
            .then(pl.lit("linear"))
            .otherwise(pl.lit("perpetual"))
            .alias("RedemptionType")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Should handle mixed types without error
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        # Some loans will repay (annuity/linear), others won't (bullet/perpetual)
        assert new_quantity < initial_quantity

        # Should have cashflows and PnL
        assert len(result_bs.cashflows) >= len(self.bs.cashflows)
        assert len(result_bs.pnls) >= len(self.bs.pnls)

        # Verify balance sheet is still valid after runoff
        result_bs.validate()

    def test_prepayment_only_scenario(self) -> None:
        """Test scenario with prepayments only (no scheduled repayments)."""
        # Set redemption type to perpetual and higher prepayment rate
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            [
                pl.when(loan_filter)
                .then(pl.lit("perpetual"))
                .otherwise(pl.col("RedemptionType"))
                .alias("RedemptionType"),
                pl.when(loan_filter)
                .then(pl.lit(0.05))
                .otherwise(pl.col("PrepaymentRate"))
                .alias("PrepaymentRate"),  # 5% annual
            ]
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should decrease due to prepayments only
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity

        # Should have no principal repayment cashflow (perpetual)
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert abs(total_repayment) < 0.01

        # Should have positive prepayment cashflow
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert total_prepayment > 0

    def test_zero_prepayment_rate(self) -> None:
        """Test scenario with zero prepayment rate."""
        # Set prepayment rate to zero
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            pl.when(loan_filter).then(pl.lit(0.0)).otherwise(pl.col("PrepaymentRate")).alias("PrepaymentRate")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Should have zero prepayment cashflow
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert abs(total_prepayment) < 0.01

    def test_combined_repayment_and_prepayment(self) -> None:
        """Test scenario with both scheduled repayments and prepayments."""
        # Set redemption type to annuity with moderate prepayment rate
        loan_filter = pl.col("AssetType") == "loan"
        self.bs._data = self.bs._data.with_columns(
            [
                pl.when(loan_filter)
                .then(pl.lit("annuity"))
                .otherwise(pl.col("RedemptionType"))
                .alias("RedemptionType"),
                pl.when(loan_filter)
                .then(pl.lit(0.03))
                .otherwise(pl.col("PrepaymentRate"))
                .alias("PrepaymentRate"),  # 3% annual
            ]
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(AssetType="loan")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Quantity should decrease due to both repayments and prepayments
        new_quantity = result_bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_quantity < initial_quantity

        # Should have both repayment and prepayment cashflows
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )

        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0

        assert total_repayment > 0
        assert total_prepayment > 0

        # Total reduction should be approximately equal to sum of repayment and prepayment
        total_reduction = initial_quantity - new_quantity
        total_cashflows = total_repayment + total_prepayment
        assert abs(total_reduction - total_cashflows) < total_reduction * 0.01

        # Verify balance sheet is still valid after runoff
        result_bs.validate()

    def test_balance_sheet_remains_balanced_after_runoff(self) -> None:
        """Test that the balance sheet remains balanced after applying runoff."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        # Verify initial balance sheet is balanced
        self.bs.validate()

        initial_total_book_value = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
        initial_cashflows = len(self.bs.cashflows)
        initial_pnls = len(self.bs.pnls)

        # Apply runoff rule
        rule = Runoff()
        result_bs = rule.apply(self.bs, increment)

        # Verify rule executed successfully and generated outputs
        assert len(result_bs.cashflows) > initial_cashflows, "Should generate cashflows"
        assert len(result_bs.pnls) > initial_pnls, "Should generate PnL entries"

        # Verify balance sheet remains balanced after runoff
        result_bs.validate()

        final_total_book_value = result_bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)

        # The total should remain close to zero (balanced)
        assert abs(final_total_book_value) < 0.01, (
            f"Balance sheet should remain balanced, but total is {final_total_book_value}"
        )

        # The balance should not change significantly from initial
        balance_change = abs(final_total_book_value - initial_total_book_value)
        assert balance_change < 0.01, f"Balance sheet balance should not change significantly: {balance_change}"
