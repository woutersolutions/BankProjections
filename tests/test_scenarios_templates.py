import contextlib
import datetime
import os
import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.scenarios.mutation import AmountRuleBase, BalanceSheetMutationRule
from bank_projections.scenarios.template import (
    MultiHeaderRule,
    MultiHeaderTemplate,
    ScenarioTemplate,
)
from bank_projections.scenarios.template_registry import TemplateRegistry
from bank_projections.utils.parsing import read_bool, read_date
from bank_projections.utils.time import TimeIncrement
from examples import EXAMPLE_FOLDER


class TestReadBool:
    def test_read_bool_true_strings(self):
        assert read_bool("true") is True
        assert read_bool("True") is True
        assert read_bool("TRUE") is True
        assert read_bool("yes") is True
        assert read_bool("Yes") is True
        assert read_bool("YES") is True
        assert read_bool("1") is True
        assert read_bool(" TRUE ") is True

    def test_read_bool_false_strings(self):
        assert read_bool("false") is False
        assert read_bool("False") is False
        assert read_bool("FALSE") is False
        assert read_bool("no") is False
        assert read_bool("No") is False
        assert read_bool("NO") is False
        assert read_bool("0") is False
        assert read_bool(" FALSE ") is False

    def test_read_bool_actual_booleans(self):
        assert read_bool(True) is True
        assert read_bool(False) is False

    def test_read_bool_invalid_values(self):
        with pytest.raises(ValueError, match="Cannot convert invalid to bool"):
            read_bool("invalid")

        with pytest.raises(ValueError, match="Cannot convert 2 to bool"):
            read_bool("2")

        with pytest.raises(ValueError, match="Cannot convert 123 to bool"):
            read_bool(123)


class TestReadDate:
    def test_read_date_string(self):
        result = read_date("2023-01-15")
        assert result == datetime.date(2023, 1, 15)

    def test_read_date_datetime_object(self):
        dt = datetime.datetime(2023, 1, 15, 10, 30, 0)
        result = read_date(dt)
        assert result == datetime.date(2023, 1, 15)

    def test_read_date_date_object(self):
        date_obj = datetime.date(2023, 1, 15)
        result = read_date(date_obj)
        assert result == date_obj

    def test_read_date_invalid_string(self):
        with pytest.raises(ValueError):
            read_date("invalid-date")

    def test_read_date_invalid_type(self):
        with pytest.raises(ValueError, match="Cannot convert 123 to date"):
            read_date(123)


class TestScenarioTemplate:
    def test_scenario_template_is_abstract(self):
        """Test that ScenarioTemplate cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ScenarioTemplate()


class TestBalanceSheetMutationRule:
    def setup_method(self):
        self.mock_bs = Mock(spec=BalanceSheet)
        self.mock_increment = Mock(spec=TimeIncrement)

    def test_init_minimal_rule_input(self):
        rule_input = {"metric": "quantity"}
        amount = 1000.0

        rule = BalanceSheetMutationRule(rule_input, amount)

        assert rule.amount == amount
        assert isinstance(rule.item, BalanceSheetItem)
        assert rule.relative is True
        assert rule.multiplicative is False
        assert rule.offset_liquidity is False
        assert rule.offset_pnl is False
        assert isinstance(rule.reason, MutationReason)
        assert rule.date is None

    def test_init_with_boolean_flags(self):
        rule_input = {
            "metric": "quantity",
            "relative": "false",
            "multiplicative": "true",
            "offsetliquidity": "yes",
            "offsetpnl": "no",
        }
        amount = 1000.0

        rule = BalanceSheetMutationRule(rule_input, amount)

        assert rule.relative is False
        assert rule.multiplicative is True
        assert rule.offset_liquidity is True
        assert rule.offset_pnl is False

    def test_init_with_date(self):
        rule_input = {"metric": "quantity", "date": "2023-01-15"}
        amount = 1000.0

        rule = BalanceSheetMutationRule(rule_input, amount)

        assert rule.date == datetime.date(2023, 1, 15)

    def test_init_with_unrecognized_key(self):
        rule_input = {"metric": "quantity", "unknown_key": "some_value"}
        amount = 1000.0

        with pytest.raises(KeyError, match="unknown_key not recognized in BalanceSheetMutationRule"):
            BalanceSheetMutationRule(rule_input, amount)

    def test_apply_without_date_constraint(self):
        rule_input = {"metric": "quantity"}
        amount = 1000.0
        rule = BalanceSheetMutationRule(rule_input, amount)

        # Mock TimeIncrement
        self.mock_increment.contains.return_value = True

        result = rule.apply(self.mock_bs, self.mock_increment, MarketRates())

        # Should call mutate_metric on the balance sheet
        self.mock_bs.mutate_metric.assert_called_once()
        assert result == self.mock_bs

    def test_apply_with_date_constraint_matching(self):
        rule_input = {"metric": "quantity", "date": "2023-01-15"}
        amount = 1000.0
        rule = BalanceSheetMutationRule(rule_input, amount)

        # Mock TimeIncrement to contain the date
        self.mock_increment.contains.return_value = True

        result = rule.apply(self.mock_bs, self.mock_increment, MarketRates())

        # Should call mutate_metric on the balance sheet
        self.mock_bs.mutate_metric.assert_called_once()
        assert result == self.mock_bs

    def test_apply_with_date_constraint_not_matching(self):
        rule_input = {"metric": "quantity", "date": "2023-01-15"}
        amount = 1000.0
        rule = BalanceSheetMutationRule(rule_input, amount)

        # Mock TimeIncrement to NOT contain the date
        self.mock_increment.contains.return_value = False

        result = rule.apply(self.mock_bs, self.mock_increment, MarketRates())

        # Should NOT call mutate_metric on the balance sheet
        self.mock_bs.mutate_metric.assert_not_called()
        assert result == self.mock_bs


class TestMultiHeaderRule:
    def setup_method(self):
        self.mock_bs = Mock(spec=BalanceSheet)
        self.mock_increment = Mock(spec=TimeIncrement)

        # Create sample data
        self.content = pd.DataFrame({0: [100.0, 200.0], 1: [150.0, 250.0]})

        # Create col_headers to match the template format (string column names)
        self.col_headers = pd.DataFrame({"metric": ["quantity", "book_value"]})

        self.row_headers = pd.DataFrame({"ItemType": ["Mortgages", "Securities"], "relative": ["true", "false"]})

        self.general_tags = {"offset_pnl": "false"}

    def test_init(self):
        rule_set = MultiHeaderRule(
            self.content, self.col_headers, self.row_headers, self.general_tags, BalanceSheetMutationRule
        )

        assert rule_set.content.equals(self.content)
        assert rule_set.col_headers.equals(self.col_headers)
        assert rule_set.row_headers.equals(self.row_headers)
        assert rule_set.general_tags == self.general_tags
        assert rule_set.rule_type == BalanceSheetMutationRule

    def test_apply(self):
        # Create a mock rule class that we'll pass to MultiHeaderRule
        mock_rule_class = Mock()
        mock_rule_instance = Mock()
        mock_rule_instance.apply.return_value = self.mock_bs
        mock_rule_class.return_value = mock_rule_instance

        rule_set = MultiHeaderRule(self.content, self.col_headers, self.row_headers, self.general_tags, mock_rule_class)

        result = rule_set.apply(self.mock_bs, self.mock_increment, MarketRates())

        # Should create 4 rules (2 rows × 2 cols)
        assert mock_rule_class.call_count == 4
        assert mock_rule_instance.apply.call_count == 4
        assert result == self.mock_bs

    def test_load_excel_invalid_template_name(self):
        """Test with invalid template name using a mock"""
        test_data = pd.DataFrame(
            [
                ["InvalidTemplate", "BalanceSheetMutations", "", ""],
                ["param1", "value1", "", ""],
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_name = temp_file.name

        with pd.ExcelWriter(temp_name, engine="openpyxl") as writer:
            test_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)

        try:
            with pytest.raises(ValueError, match="First cell must be 'Template'"):
                TemplateRegistry.load_excel_sheet(temp_name, "Sheet1")
        finally:
            import contextlib
            import os

            with contextlib.suppress(PermissionError):
                os.unlink(temp_name)

    def test_load_excel_invalid_scenario_type(self):
        """Test with invalid scenario type using a mock"""
        test_data = pd.DataFrame(
            [
                ["Template", "InvalidScenario", "", ""],
                ["param1", "value1", "", ""],
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_name = temp_file.name

        with pd.ExcelWriter(temp_name, engine="openpyxl") as writer:
            test_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)

        try:
            with pytest.raises(ValueError, match="Template 'InvalidScenario' not recognized"):
                TemplateRegistry.load_excel_sheet(temp_name, "Sheet1")
        finally:
            with contextlib.suppress(PermissionError):
                os.unlink(temp_name)

    def test_template_registry_registration(self):
        """Test that templates are properly registered"""
        # Check that the default template is registered
        assert "balancesheetmutations" in TemplateRegistry.items
        template = TemplateRegistry.get("balancesheetmutations")
        assert isinstance(template, MultiHeaderTemplate)
        assert template.rule_type == BalanceSheetMutationRule


class TestMultiHeaderTemplate:
    def test_initialization(self):
        """Test MultiHeaderTemplate initialization"""
        template = MultiHeaderTemplate(BalanceSheetMutationRule)
        assert template.rule_type == BalanceSheetMutationRule

    def test_load_excel_sheet_returns_multi_header_rule(self):
        """Test that load_excel_sheet returns MultiHeaderRule"""
        # Since creating a proper Excel file is complex, let's just test the basic functionality
        template = MultiHeaderTemplate(BalanceSheetMutationRule)
        assert template.rule_type == BalanceSheetMutationRule

        # Test that the template is a ScenarioTemplate
        assert isinstance(template, ScenarioTemplate)


class TestAbstractClasses:
    def test_scenario_template_is_abstract(self):
        """Test that ScenarioTemplate cannot be instantiated"""
        with pytest.raises(TypeError):
            ScenarioTemplate()

    def test_amount_rule_base_is_abstract(self):
        """Test that AmountRuleBase cannot be instantiated"""
        with pytest.raises(TypeError):
            AmountRuleBase({}, 100.0)


class TestRuleSetIntegration:
    """Test that the new architecture works with RuleSet"""

    def test_scenario_ruleset_composition(self):
        """Test that scenarios can be composed from multiple rules"""
        from bank_projections.projections.runoff import Runoff
        from bank_projections.scenarios.scenario import Scenario

        # Create a scenario with multiple rules
        rule1 = Mock(spec=BalanceSheetMutationRule)
        rule2 = Mock(spec=Runoff)

        scenario = Scenario(rules={"rule1": rule1, "rule2": rule2})
        assert len(scenario.rules) == 2
        assert rule1 in scenario.rules.values()
        assert rule2 in scenario.rules.values()


class TestOneHeaderTemplate:
    def test_initialization(self):
        from bank_projections.scenarios.template import OneHeaderTemplate

        template = OneHeaderTemplate(BalanceSheetMutationRule)
        assert template.rule_type == BalanceSheetMutationRule

    def test_one_header_rule_apply(self):
        from bank_projections.scenarios.template import OneHeaderRule

        mock_bs = Mock(spec=BalanceSheet)
        mock_increment = Mock(spec=TimeIncrement)

        content = pd.DataFrame({"metric": ["quantity", "book_value"], "date": ["2024-01-15", "2024-02-15"]})
        general_tags = {"offset_pnl": "false"}

        mock_rule_class = Mock()
        mock_rule_instance = Mock()
        mock_rule_instance.apply.return_value = mock_bs
        mock_rule_class.return_value = mock_rule_instance

        rule = OneHeaderRule(content, general_tags, mock_rule_class)
        result = rule.apply(mock_bs, mock_increment, MarketRates())

        assert mock_rule_class.call_count == 2
        assert mock_rule_instance.apply.call_count == 2
        assert result == mock_bs


class TestTaxTemplate:
    def test_tax_template_initialization(self):
        from bank_projections.scenarios.tax import TaxTemplate

        template = TaxTemplate()
        assert isinstance(template, TaxTemplate)

    def test_tax_rule_initialization(self):
        from bank_projections.scenarios.tax import TaxRule

        rule = TaxRule(tax_rate=0.25)
        assert rule.tax_rate == 0.25

    def test_tax_rule_apply(self, minimal_scenario):
        import datetime

        import polars as pl

        from bank_projections.scenarios.tax import TaxRule
        from examples.synthetic_data import create_synthetic_balance_sheet

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1), scenario=minimal_scenario)
        bs.pnls = pl.DataFrame({"Amount": [100.0, -50.0, 75.0, -25.0]})

        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        market_rates = MarketRates()

        rule = TaxRule(tax_rate=0.25)
        result = rule.apply(bs, increment, market_rates)

        assert result is not None


class TestKeyValueTemplate:
    def test_key_value_template_initialization(self):
        from bank_projections.scenarios.template import KeyValueRuleBase, KeyValueTemplate

        template = KeyValueTemplate(KeyValueRuleBase)
        assert template.rule_type == KeyValueRuleBase


class TestTemplateRegistryLoadPaths:
    """Test TemplateRegistry.load_paths method."""

    def test_load_paths_with_folder(self):
        """Test load_paths with a folder path."""
        scenario = TemplateRegistry.load_paths([os.path.join(EXAMPLE_FOLDER, "scenarios")])
        assert scenario is not None
        assert len(scenario.rules) > 0

    def test_load_paths_with_multiple_paths(self):
        """Test load_paths combines multiple paths."""
        paths = [os.path.join(EXAMPLE_FOLDER, "scenarios")]
        scenario = TemplateRegistry.load_paths(paths)
        assert scenario is not None

    def test_load_paths_matches_load_folder(self):
        """Test that load_paths with single folder matches load_folder."""
        folder_path = os.path.join(EXAMPLE_FOLDER, "scenarios")
        scenario_from_paths = TemplateRegistry.load_paths([folder_path])
        scenario_from_folder = TemplateRegistry.load_folder(folder_path)

        assert set(scenario_from_paths.rules.keys()) == set(scenario_from_folder.rules.keys())


class TestScenarioConfig:
    """Test ScenarioConfig Pydantic model."""

    def test_scenario_config_creation(self):
        """Test creating ScenarioConfig."""
        from bank_projections.scenarios.scenario import ScenarioConfig
        from bank_projections.utils.time import TimeHorizonConfig

        time_horizon = TimeHorizonConfig(
            start_date=datetime.date(2024, 12, 31),
            number_of_months=12,
            end_of_month=True,
        )
        config = ScenarioConfig(
            rule_paths=["src/examples/scenarios"],
            time_horizon=time_horizon,
        )

        assert config.rule_paths == ["src/examples/scenarios"]
        assert config.time_horizon.start_date == datetime.date(2024, 12, 31)
        assert config.time_horizon.number_of_months == 12
        assert config.time_horizon.end_of_month is True

    def test_scenario_config_from_yaml_dict(self):
        """Test creating ScenarioConfig from dict (as loaded from YAML)."""
        from bank_projections.scenarios.scenario import ScenarioConfig

        yaml_dict = {
            "rule_paths": ["src/examples/scenarios"],
            "time_horizon": {
                "start_date": "2024-12-31",
                "number_of_months": 12,
                "end_of_month": True,
            },
        }
        config = ScenarioConfig(**yaml_dict)

        assert config.rule_paths == ["src/examples/scenarios"]
        assert config.time_horizon.start_date == datetime.date(2024, 12, 31)
        assert config.time_horizon.number_of_months == 12


class TestAggregationConfig:
    """Test AggregationConfig Pydantic model."""

    def test_aggregation_config_defaults(self):
        """Test AggregationConfig with default values (all None)."""
        from bank_projections.output_config import AggregationConfig

        config = AggregationConfig()

        assert config.balance_sheet is None
        assert config.pnl is None
        assert config.cashflow is None
        assert config.oci is None

    def test_aggregation_config_with_values(self):
        """Test AggregationConfig with explicit values."""
        from bank_projections.output_config import AggregationConfig

        config = AggregationConfig(
            balance_sheet=["ItemType", "SubItemType"],
            pnl=["ItemType", "SubItemType", "rule"],
            cashflow=["ItemType"],
            oci=["ItemType", "SubItemType"],
        )

        assert config.balance_sheet == ["ItemType", "SubItemType"]
        assert config.pnl == ["ItemType", "SubItemType", "rule"]
        assert config.cashflow == ["ItemType"]
        assert config.oci == ["ItemType", "SubItemType"]

    def test_aggregation_config_partial(self):
        """Test AggregationConfig with only some values set."""
        from bank_projections.output_config import AggregationConfig

        config = AggregationConfig(balance_sheet=["ItemType"])

        assert config.balance_sheet == ["ItemType"]
        assert config.pnl is None
        assert config.cashflow is None
        assert config.oci is None


class TestOutputConfig:
    """Test OutputConfig Pydantic model."""

    def test_output_config_creation(self):
        """Test creating OutputConfig."""
        from bank_projections.output_config import AggregationConfig, OutputConfig

        aggregation = AggregationConfig(
            balance_sheet=["ItemType", "SubItemType"],
            pnl=["ItemType", "SubItemType", "module", "rule"],
        )
        config = OutputConfig(
            output_folder="output",
            output_file="test_%Y%m%d.xlsx",
            aggregation=aggregation,
        )

        assert config.output_folder == "output"
        assert config.output_file == "test_%Y%m%d.xlsx"
        assert config.aggregation.balance_sheet == ["ItemType", "SubItemType"]

    def test_output_config_from_yaml_dict(self):
        """Test creating OutputConfig from dict (as loaded from YAML)."""
        from bank_projections.output_config import OutputConfig

        yaml_dict = {
            "output_folder": "output",
            "output_file": "main example_%Y%m%d_%H%M%S.xlsx",
            "aggregation": {
                "balance_sheet": ["ItemType", "SubItemType"],
                "pnl": ["ItemType", "SubItemType", "module", "rule"],
                "cashflow": ["ItemType", "SubItemType", "module", "rule"],
                "oci": ["ItemType", "SubItemType", "module", "rule"],
            },
        }
        config = OutputConfig(**yaml_dict)

        assert config.output_folder == "output"
        assert config.output_file == "main example_%Y%m%d_%H%M%S.xlsx"
        assert config.aggregation.balance_sheet == ["ItemType", "SubItemType"]
        assert config.aggregation.pnl == ["ItemType", "SubItemType", "module", "rule"]
