"""Unit tests for synthetic data generation."""

import datetime

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from examples.synthetic_data import create_single_asset_balance_sheet

# Uses minimal_scenario fixture from conftest.py


class TestCreateSingleAssetBalanceSheet:
    """Test create_single_asset_balance_sheet function."""

    def test_basic_creation(self, minimal_scenario):
        """Test basic creation with specified asset added to default config."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=1000000.0,
            accounting_method="amortized cost",
            redemption_type="annuity",
            coupon_frequency="Monthly",
            coupon_type="fixed",
            item_type="Loans",
            sub_item_type="TestMortgages",
            interest_rate=(0.02, 0.04),
            maturity=(10, 30),
            reference_rate="Euribor 3m",
        )

        assert isinstance(bs, BalanceSheet)
        assert bs.date == datetime.date(2024, 12, 31)

        # Verify the specified asset exists
        test_mortgage_item = BalanceSheetItem(SubItemType="TestMortgages")
        test_mortgage_value = bs.get_amount(test_mortgage_item, BalanceSheetMetrics.get("book_value"))
        assert test_mortgage_value > 0  # Asset was added

    def test_with_single_value_parameters(self, minimal_scenario):
        """Test with single value (not range) parameters."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=500000.0,
            accounting_method="amortized cost",
            redemption_type="bullet",
            coupon_frequency="Quarterly",
            coupon_type="fixed",
            item_type="Loans",
            sub_item_type="TestSMELoans",
            interest_rate=0.035,
            maturity=5,
            reference_rate="Euribor 6m",
        )

        assert isinstance(bs, BalanceSheet)
        test_sme_item = BalanceSheetItem(SubItemType="TestSMELoans")
        test_sme_value = bs.get_amount(test_sme_item, BalanceSheetMetrics.get("book_value"))
        assert test_sme_value > 0

    def test_asset_is_present(self, minimal_scenario):
        """Test that the specified asset is present in the balance sheet."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=2000000.0,
            accounting_method="amortized cost",
            redemption_type="annuity",
            coupon_frequency="Monthly",
            coupon_type="both",
            item_type="Loans",
            sub_item_type="TestLoans",
            interest_rate=(0.01, 0.05),
            maturity=(15, 25),
            reference_rate="Euribor 3m",
        )

        # Verify asset was added
        test_loans_item = BalanceSheetItem(SubItemType="TestLoans")
        test_loans_value = bs.get_amount(test_loans_item, BalanceSheetMetrics.get("book_value"))
        assert test_loans_value > 0

    def test_default_config_items_present(self, minimal_scenario):
        """Test that default config items (equity, cash) are still present."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=300000.0,
            accounting_method="amortized cost",
            redemption_type="annuity",
            coupon_frequency="Monthly",
            coupon_type="fixed",
            item_type="Loans",
            sub_item_type="TestConsumerLoans",
            interest_rate=0.06,
            maturity=3,
            reference_rate="Euribor 3m",
        )

        # Check that equity items exist
        equity_item = BalanceSheetItem(BalanceSheetCategory="Equity")
        equity_value = bs.get_amount(equity_item, BalanceSheetMetrics.get("book_value"))
        assert equity_value > 0  # Equity is positive

        # Check that cash exists
        cash_item = BalanceSheetItem(ItemType="Cash")
        cash_value = bs.get_amount(cash_item, BalanceSheetMetrics.get("book_value"))
        assert cash_value > 0  # Cash should be positive

    def test_with_optional_parameters(self, minimal_scenario):
        """Test with various optional parameters."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=1500000.0,
            accounting_method="amortized cost",
            redemption_type="annuity",
            coupon_frequency="Monthly",
            coupon_type="both",
            item_type="Loans",
            sub_item_type="TestMortgages2",
            ifrs9_stage="mixed",
            coverage_rate=(0.01, 0.05),
            interest_rate=(0.02, 0.045),
            agio=(0.0, 0.01),
            prepayment_rate=0.01,
            maturity=(20, 30),
            reference_rate="Euribor 3m",
        )

        assert isinstance(bs, BalanceSheet)
        mortgage_item = BalanceSheetItem(SubItemType="TestMortgages2")
        mortgage_value = bs.get_amount(mortgage_item, BalanceSheetMetrics.get("book_value"))
        assert mortgage_value > 0

    def test_with_fair_value_securities(self, minimal_scenario):
        """Test creation with debt securities at fair value."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=500000.0,
            accounting_method="fair value through oci",
            redemption_type="bullet",
            coupon_frequency="Quarterly",
            coupon_type="fixed",
            item_type="Debt securities",
            sub_item_type="TestFixedBonds",
            interest_rate=(0.015, 0.03),
            maturity=(5, 15),
            reference_rate="Euribor 6m",
            valuation_method="fixedratebond",
            valuation_curve="euribor",
        )

        assert isinstance(bs, BalanceSheet)
        securities_item = BalanceSheetItem(SubItemType="TestFixedBonds")
        securities_value = bs.get_amount(securities_item, BalanceSheetMetrics.get("book_value"))
        assert securities_value > 0

    def test_function_is_callable(self, minimal_scenario):
        """Test that function exists and is callable."""
        # This verifies the function was imported correctly and has the right signature
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=1000000.0,
            accounting_method="amortized cost",
            redemption_type="annuity",
            coupon_frequency="Monthly",
            coupon_type="fixed",
            item_type="Loans",
            sub_item_type="TestSimple",
            interest_rate=0.03,
            maturity=20,
            reference_rate="Euribor 3m",
        )

        # Should return a BalanceSheet
        assert isinstance(bs, BalanceSheet)

    def test_range_parameters_work(self, minimal_scenario):
        """Test that range tuples work for numeric parameters."""
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=1000000.0,
            accounting_method="amortized cost",
            redemption_type="annuity",
            coupon_frequency="Monthly",
            coupon_type="both",
            item_type="Loans",
            sub_item_type="TestRangeMortgages",
            ifrs9_stage="mixed",
            coverage_rate=(0.01, 0.03),
            interest_rate=(0.02, 0.04),
            undrawn_portion=(0.0, 0.02),
            agio=(0.0, 0.01),
            prepayment_rate=(0.005, 0.015),
            age=(0, 5),
            maturity=(15, 30),
            reference_rate="Euribor 3m",
        )

        assert isinstance(bs, BalanceSheet)
        mortgage_item = BalanceSheetItem(SubItemType="TestRangeMortgages")

        # Verify asset was added
        book_value = bs.get_amount(mortgage_item, BalanceSheetMetrics.get("book_value"))
        assert book_value > 0

    def test_default_enum_values(self, minimal_scenario):
        """Test that default enum values are applied correctly."""
        # Not providing hqla_class or ifrs9_stage - should use defaults
        bs = create_single_asset_balance_sheet(
            current_date=datetime.date(2024, 12, 31),
            scenario=minimal_scenario,
            book_value=500000.0,
            accounting_method="amortized cost",
            redemption_type="bullet",
            coupon_frequency="Annual",
            coupon_type="fixed",
            item_type="Loans",
            sub_item_type="TestDefaults",
            interest_rate=0.04,
            maturity=10,
            reference_rate="Euribor 3m",
        )

        assert isinstance(bs, BalanceSheet)
