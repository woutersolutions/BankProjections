"""Tests for the application configuration system."""

import tempfile
from pathlib import Path

from bank_projections.app_config import (
    AppConfig,
    ClassificationConfig,
    Config,
    DictionaryEntry,
    LabelsConfig,
    get_config,
    init_config,
    is_config_initialized,
    reset_config,
)


class TestLabelsConfig:
    """Test LabelsConfig model."""

    def test_labels_config_creation(self) -> None:
        """Test creating LabelsConfig."""
        config = LabelsConfig(
            cashflow=["ItemType", "module"],
            pnl=["ItemType", "rule"],
            oci=["ItemType"],
        )

        assert config.cashflow == ["ItemType", "module"]
        assert config.pnl == ["ItemType", "rule"]
        assert config.oci == ["ItemType"]


class TestDictionaryEntry:
    """Test DictionaryEntry model."""

    def test_dictionary_entry_creation(self) -> None:
        """Test creating DictionaryEntry."""
        entry = DictionaryEntry(
            keyword="Nominal",
            data_type="money",
            required=True,
            metric_type="StoredAmount",
            registry=None,
            description="Principal amount",
        )
        assert entry.keyword == "Nominal"
        assert entry.data_type == "money"
        assert entry.required is True
        assert entry.metric_type == "StoredAmount"
        assert entry.registry is None

    def test_dictionary_entry_required_parsing(self) -> None:
        """Test required field parsing from string."""
        entry = DictionaryEntry.model_validate({
            "keyword": "Test",
            "data_type": "float",
            "required": "yes",
            "metric_type": "StoredWeight",
            "registry": "",
            "description": "Test field",
        })
        assert entry.required is True

        entry2 = DictionaryEntry.model_validate({
            "keyword": "Test2",
            "data_type": "float",
            "required": "no",
            "metric_type": "StoredWeight",
            "registry": "",
            "description": "Test field 2",
        })
        assert entry2.required is False


class TestClassificationConfig:
    """Test ClassificationConfig model."""

    def test_classification_config_creation(self) -> None:
        """Test creating ClassificationConfig."""
        config = ClassificationConfig(
            column_name="Book",
            registry_import="bank_projections.projections.book.BookRegistry",
        )

        assert config.column_name == "Book"
        assert config.registry_import == "bank_projections.projections.book.BookRegistry"

    def test_classification_config_get_registry(self) -> None:
        """Test dynamically importing a registry class."""
        config = ClassificationConfig(
            column_name="Book",
            registry_import="bank_projections.projections.book.BookRegistry",
        )

        registry = config.get_registry()

        # Should be a class with registry methods
        assert hasattr(registry, "items")
        assert hasattr(registry, "get")


class TestAppConfig:
    """Test AppConfig model."""

    def test_app_config_creation(self) -> None:
        """Test creating AppConfig from dict structure."""
        dictionary = [
            DictionaryEntry(
                keyword="ItemType", data_type="string", required=True,
                metric_type="Label", registry=None, description="Primary classification"
            ),
            DictionaryEntry(
                keyword="Book", data_type="enum", required=True,
                metric_type="Classification",
                registry="bank_projections.projections.book.BookRegistry",
                description="Trading/Banking book"
            ),
        ]
        config_dict = {
            "labels": {
                "cashflow": ["ItemType", "module"],
                "pnl": ["ItemType", "rule"],
                "oci": ["ItemType"],
            },
            "profitability_outlooks": ["Monthly", "Annual"],
            "dictionary": dictionary,
        }

        config = AppConfig(**config_dict)

        assert config.balance_sheet_labels() == ["ItemType"]
        assert config.profitability_outlooks == ["Monthly", "Annual"]

    def test_app_config_get_classifications(self) -> None:
        """Test getting classifications dict with imported registries."""
        dictionary = [
            DictionaryEntry(
                keyword="Book", data_type="enum", required=True,
                metric_type="Classification",
                registry="bank_projections.projections.book.BookRegistry",
                description="Trading/Banking book"
            ),
            DictionaryEntry(
                keyword="HQLAClass", data_type="enum", required=False,
                metric_type="Classification",
                registry="bank_projections.financials.hqla_class.HQLARegistry",
                description="HQLA classification"
            ),
        ]
        config_dict = {
            "labels": {
                "cashflow": ["ItemType"],
                "pnl": ["ItemType"],
                "oci": ["ItemType"],
            },
            "profitability_outlooks": ["Monthly"],
            "dictionary": dictionary,
        }

        config = AppConfig(**config_dict)
        classifications = config.get_classifications()

        assert "Book" in classifications
        assert "HQLAClass" in classifications
        # Should be cached
        assert config.get_classifications() is classifications

    def test_app_config_label_columns(self) -> None:
        """Test label_columns combines all relevant columns."""
        dictionary = [
            DictionaryEntry(
                keyword="ItemType", data_type="string", required=True,
                metric_type="Label", registry=None, description="Primary classification"
            ),
            DictionaryEntry(
                keyword="SubItemType", data_type="string", required=True,
                metric_type="Label", registry=None, description="Secondary classification"
            ),
            DictionaryEntry(
                keyword="MaturityDate", data_type="date", required=False,
                metric_type="DateColumn", registry=None, description="Maturity date"
            ),
            DictionaryEntry(
                keyword="OriginationDate", data_type="date", required=False,
                metric_type="DateColumn", registry=None, description="Origination date"
            ),
            DictionaryEntry(
                keyword="Book", data_type="enum", required=True,
                metric_type="Classification",
                registry="bank_projections.projections.book.BookRegistry",
                description="Trading/Banking book"
            ),
        ]
        config_dict = {
            "labels": {
                "cashflow": ["ItemType"],
                "pnl": ["ItemType"],
                "oci": ["ItemType"],
            },
            "profitability_outlooks": ["Monthly"],
            "dictionary": dictionary,
        }

        config = AppConfig(**config_dict)
        label_cols = config.label_columns()

        assert "ItemType" in label_cols
        assert "SubItemType" in label_cols
        assert "MaturityDate" in label_cols
        assert "OriginationDate" in label_cols
        assert "Book" in label_cols


class TestConfigSingleton:
    """Test the configuration singleton pattern."""

    def test_get_config_auto_initializes(self) -> None:
        """Test that get_config auto-initializes from default path."""
        # Reset to ensure fresh state, then call get_config
        reset_config()
        assert not is_config_initialized()

        config = get_config()

        assert is_config_initialized()
        assert config is not None
        assert isinstance(config, AppConfig)

    def test_init_config_with_custom_path(self) -> None:
        """Test initializing config from a custom path."""
        reset_config()

        yaml_content = """
labels:
  cashflow: [CustomCF]
  pnl: [CustomPnL]
  oci: [CustomOCI]

profitability_outlooks: [CustomOutlook]
"""
        csv_content = """keyword,data_type,required,metric_type,registry,description
CustomLabel,string,yes,Label,,Custom label field
CustomDate,date,no,DateColumn,,Custom date field
CustomClass,enum,yes,Classification,bank_projections.projections.book.BookRegistry,Custom class
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as yaml_f:
            yaml_f.write(yaml_content)
            yaml_f.flush()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
                csv_f.write(csv_content)
                csv_f.flush()

                config = init_config(yaml_f.name, csv_f.name)

                assert config.balance_sheet_labels() == ["CustomLabel"]
                assert config.profitability_outlooks == ["CustomOutlook"]

            # Cleanup
            Path(csv_f.name).unlink()

        Path(yaml_f.name).unlink()
        reset_config()

    def test_reset_config(self) -> None:
        """Test resetting the config singleton."""
        # Ensure initialized
        get_config()
        assert is_config_initialized()

        reset_config()

        assert not is_config_initialized()


class TestConfigCompatibility:
    """Test the Config compatibility layer."""

    def test_config_balance_sheet_labels(self) -> None:
        """Test accessing BALANCE_SHEET_LABELS through Config class."""
        labels = Config.balance_sheet_labels()

        assert isinstance(labels, list)
        assert "ItemType" in labels

    def test_config_classifications(self) -> None:
        """Test accessing CLASSIFICATIONS through Config class."""
        classifications = Config.get_classifications()

        assert isinstance(classifications, dict)
        assert "Book" in classifications

    def test_config_label_columns(self) -> None:
        """Test Config.label_columns() method."""
        label_cols = Config.label_columns()

        assert isinstance(label_cols, list)
        assert "ItemType" in label_cols

    def test_config_required_columns(self) -> None:
        """Test Config.required_columns() method."""
        required = Config.required_columns()

        assert isinstance(required, list)
        # Should include balance sheet labels, classifications, metrics, and date columns
        assert "ItemType" in required
        assert "Book" in required
        assert "Nominal" in required  # From metrics

    def test_config_non_null_columns(self) -> None:
        """Test Config.non_null_columns() method."""
        non_null = Config.non_null_columns()

        assert isinstance(non_null, list)
        # Non-null columns are now determined by required=True in dictionary
        assert "Nominal" in non_null
        assert "ItemType" in non_null
        assert "BalanceSheetCategory" in non_null
