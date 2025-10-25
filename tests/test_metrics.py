import polars as pl
import pytest

from bank_projections.financials.balance_sheet_metrics import (
    BalanceSheetMetric,
    BaselExposure,
    BookValue,
    DerivedMetric,
    DirtyPrice,
    StoredAmount,
    StoredColumn,
    StoredWeight,
)


class TestBalanceSheetMetric:
    def test_balance_sheet_metric_is_abstract(self):
        """Test that BalanceSheetMetric cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BalanceSheetMetric()


class TestStoredColumn:
    def test_stored_column_is_abstract(self):
        """Test that StoredColumn cannot be instantiated directly"""
        with pytest.raises(TypeError):
            StoredColumn("test")


class TestStoredAmount:
    def test_stored_amount_initialization(self):
        metric = StoredAmount("TestColumn")
        assert metric.column == "TestColumn"

    def test_get_expression(self):
        metric = StoredAmount("TestColumn")
        expr = metric.get_expression
        assert isinstance(expr, pl.Expr)

    def test_set_expression(self):
        metric = StoredAmount("TestColumn")
        amount_value = 100.0
        result = metric.set_expression(amount_value)
        assert isinstance(result, pl.Expr)

    def test_aggregation_expression(self):
        metric = StoredAmount("TestColumn")
        expr = metric.aggregation_expression
        assert isinstance(expr, pl.Expr)

    def test_mutation_expression(self):
        metric = StoredAmount("TestColumn")
        filter_expr = pl.col("AssetType") == "Mortgages"
        result = metric.mutation_expression(100.0, filter_expr)
        assert isinstance(result, pl.Expr)

    def test_mutation_column(self):
        metric = StoredAmount("TestColumn")
        assert metric.mutation_column == "TestColumn"


class TestStoredWeight:
    def test_stored_weight_initialization_default(self):
        metric = StoredWeight("TestColumn")
        assert metric.column == "TestColumn"

    def test_stored_weight_initialization_with_allocation(self):
        weight_expr = pl.col("CustomWeight")
        metric = StoredWeight("TestColumn", weight_expr)
        assert metric.column == "TestColumn"

    def test_get_expression(self):
        metric = StoredWeight("TestColumn")
        expr = metric.get_expression
        assert isinstance(expr, pl.Expr)

    def test_set_expression(self):
        metric = StoredWeight("TestColumn")
        amount_value = 0.05  # 5% weight
        result = metric.set_expression(amount_value)
        assert isinstance(result, pl.Expr)

    def test_aggregation_expression(self):
        metric = StoredWeight("TestColumn")
        expr = metric.aggregation_expression
        assert isinstance(expr, pl.Expr)

    def test_mutation_expression(self):
        metric = StoredWeight("TestColumn")
        filter_expr = pl.col("AssetType") == "Mortgages"
        result = metric.mutation_expression(0.05, filter_expr)
        assert isinstance(result, pl.Expr)

    def test_mutation_column(self):
        metric = StoredWeight("TestColumn")
        assert metric.mutation_column == "TestColumn"


class TestDerivedMetric:
    def test_derived_metric_is_abstract(self):
        """Test that DerivedMetric cannot be instantiated directly"""
        with pytest.raises(TypeError):
            DerivedMetric()

    def test_derived_metric_mutation_expression_raises_error(self):
        """Test that mutation_expression raises NotImplementedError"""
        metric = DirtyPrice()
        filter_expr = pl.col("AssetType") == "Mortgages"

        with pytest.raises(NotImplementedError, match="Derived metric cannot be modified"):
            metric.mutation_expression(100.0, filter_expr)

    def test_derived_metric_mutation_column_raises_error(self):
        """Test that mutation_column raises NotImplementedError"""
        metric = DirtyPrice()

        with pytest.raises(NotImplementedError, match="Derived metric cannot be modified"):
            _ = metric.mutation_column


class TestDirtyPrice:
    def test_dirty_price_initialization(self):
        metric = DirtyPrice()

    def test_get_expression(self):
        metric = DirtyPrice()
        expr = metric.get_expression
        assert isinstance(expr, pl.Expr)

    def test_aggregation_expression(self):
        metric = DirtyPrice()
        expr = metric.aggregation_expression
        assert isinstance(expr, pl.Expr)


class TestExposure:
    def test_exposure_initialization(self):
        metric = BaselExposure()

    def test_get_expression(self):
        metric = BaselExposure()
        expr = metric.get_expression
        assert isinstance(expr, pl.Expr)

    def test_aggregation_expression(self):
        metric = BaselExposure()
        expr = metric.aggregation_expression
        assert isinstance(expr, pl.Expr)


class TestBookValue:
    def test_book_value_initialization(self):
        metric = BookValue()

    def test_get_expression(self):
        metric = BookValue()
        expr = metric.get_expression
        assert isinstance(expr, pl.Expr)

    def test_aggregation_expression(self):
        metric = BookValue()
        expr = metric.aggregation_expression
        assert isinstance(expr, pl.Expr)


class TestMetricIntegration:
    """Integration tests to verify metrics work with actual data"""

    def test_stored_amount_with_real_data(self):
        # Create test data
        df = pl.DataFrame({"Quantity": [1000.0, 2000.0, 1500.0], "AssetType": ["Mortgages", "Securities", "Mortgages"]})

        metric = StoredAmount("Quantity")

        # Test get_expression
        result = df.select(metric.get_expression.alias("result"))
        assert result["result"].to_list() == [1000.0, 2000.0, 1500.0]

        # Test aggregation_expression
        result = df.select(metric.aggregation_expression.alias("total"))
        assert result["total"].item() == 4500.0

    def test_dirty_price_with_real_data(self):
        # Create test data
        df = pl.DataFrame({"Quantity": [1000.0, 2000.0], "CleanPrice": [100.0, 95.0], "AccruedInterest": [5.0, 10.0]})

        metric = DirtyPrice()

        # Test get_expression (clean_price + accrued_interest/quantity = price per unit)
        result = df.select(metric.get_expression.alias("dirty_price"))
        expected = [100.005, 95.005]  # (100 + 5/1000), (95 + 10/2000)
        assert result["dirty_price"].to_list() == expected

    def test_exposure_with_real_data(self):
        # Create test data with all required columns for BaselExposure
        df = pl.DataFrame({
            "Quantity": [1000.0, 2000.0],
            "AccountingMethod": ["amortizedcost", "amortizedcost"],
            "Agio": [10.0, 20.0],
            "AccruedInterest": [5.0, 10.0],
            "CleanPrice": [1.0, 1.0],  # Not used for AC but required
            "CCF": [0.5, 0.5],
            "Undrawn": [100.0, 200.0],
            "OtherOffBalanceWeight": [0.0, 0.0],
        })

        metric = BaselExposure()

        # Test get_expression
        # OnBalanceExposure (AC) = Quantity + Agio + AccruedInterest = 1000+10+5=1015, 2000+20+10=2030
        # OffBalanceExposure = CCF * Undrawn + OtherOffBalanceWeight * Quantity = 0.5*100+0=50, 0.5*200+0=100
        # Total = 1015+50=1065, 2030+100=2130
        result = df.select(metric.get_expression.alias("exposure"))
        expected = [1065.0, 2130.0]
        assert result["exposure"].to_list() == expected
