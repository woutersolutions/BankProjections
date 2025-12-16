"""Tests for scenario templates and utility functions."""

import datetime
import os
from unittest.mock import Mock

import pandas as pd
import pytest

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.scenarios.excel_sheet_format import (
    MultiHeaderTableInput,
    MultiHeaderTemplate,
    TemplateType,
    TemplateTypeRegistry,
)
from bank_projections.scenarios.mutation import BalanceSheetMutationRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
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


class TestTemplateType:
    def test_template_type_is_abstract(self):
        """Test that TemplateType cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TemplateType()


class TestBalanceSheetMutationRule:
    """Test BalanceSheetMutationRule class."""

    def test_instantiation(self):
        """Test that BalanceSheetMutationRule can be instantiated without arguments."""
        rule = BalanceSheetMutationRule()
        assert rule is not None

    def test_apply_with_no_mutations(self, minimal_scenario, minimal_scenario_snapshot):
        """Test applying rule when scenario has no mutations."""
        from examples.synthetic_data import create_synthetic_balance_sheet

        rule = BalanceSheetMutationRule()
        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1), scenario=minimal_scenario)
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))

        # minimal_scenario_snapshot has no mutations
        result = rule.apply(bs, increment, minimal_scenario_snapshot)

        assert result is not None
        assert result == bs  # No changes since no mutations


class TestMultiHeaderRule:
    def setup_method(self):
        self.mock_bs = Mock(spec=BalanceSheet)
        self.mock_increment = Mock(spec=TimeIncrement)
        self.mock_scenario = Mock(spec=ScenarioSnapShot)

        # Create sample data
        self.content = pd.DataFrame({0: [100.0, 200.0], 1: [150.0, 250.0]})

        # Create col_headers to match the template format (string column names)
        self.col_headers = pd.DataFrame({"metric": ["nominal", "book_value"]})

        self.row_headers = pd.DataFrame({"ItemType": ["Mortgages", "Securities"], "relative": ["true", "false"]})

        self.general_tags = {"offset_pnl": "false"}

    def test_init(self):
        rule_set = MultiHeaderTableInput(
            content=self.content,
            col_headers=self.col_headers,
            row_headers=self.row_headers,
            general_tags=self.general_tags,
            template_name="balancesheetmutations",
        )

        assert rule_set.general_tags == self.general_tags


class TestMultiHeaderTemplate:
    """Test MultiHeaderTemplate class methods."""

    def test_is_subclass_of_template_type(self):
        """Test that MultiHeaderTemplate is a subclass of TemplateType."""
        assert issubclass(MultiHeaderTemplate, TemplateType)

    def test_matches_method_exists(self):
        """Test that MultiHeaderTemplate has a matches class method."""
        assert hasattr(MultiHeaderTemplate, "matches")
        assert callable(MultiHeaderTemplate.matches)

    def test_load_method_exists(self):
        """Test that MultiHeaderTemplate has a load class method."""
        assert hasattr(MultiHeaderTemplate, "load")
        assert callable(MultiHeaderTemplate.load)


class TestAbstractClasses:
    def test_template_type_is_abstract(self):
        """Test that TemplateType cannot be instantiated"""
        with pytest.raises(TypeError):
            TemplateType()


class TestTemplateTypeRegistryLoadFolder:
    """Test TemplateTypeRegistry.load_folder method."""

    def test_load_folder_returns_excel_inputs(self):
        """Test load_folder with a folder path returns list of ExcelInput."""
        excel_inputs = TemplateTypeRegistry.load_folder(os.path.join(EXAMPLE_FOLDER, "scenarios"))
        assert excel_inputs is not None
        assert isinstance(excel_inputs, list)
        assert len(excel_inputs) > 0


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
            input_paths=["src/examples/scenarios"],
            time_horizon=time_horizon,
        )

        assert config.input_paths == ["src/examples/scenarios"]
        assert config.time_horizon.start_date == datetime.date(2024, 12, 31)
        assert config.time_horizon.number_of_months == 12
        assert config.time_horizon.end_of_month is True


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
