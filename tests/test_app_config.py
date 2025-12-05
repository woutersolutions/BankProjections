"""Tests for the application configuration system."""

import tempfile
from pathlib import Path

from bank_projections.app_config import (
    AppConfig,
    ClassificationConfig,
    Config,
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
            balance_sheet=["ItemType", "SubItemType"],
            cashflow=["ItemType", "module"],
            pnl=["ItemType", "rule"],
            oci=["ItemType"],
            date_columns=["MaturityDate"],
        )

        assert config.balance_sheet == ["ItemType", "SubItemType"]
        assert config.cashflow == ["ItemType", "module"]
        assert config.pnl == ["ItemType", "rule"]
        assert config.oci == ["ItemType"]
        assert config.date_columns == ["MaturityDate"]


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
        config_dict = {
            "labels": {
                "balance_sheet": ["ItemType"],
                "cashflow": ["ItemType", "module"],
                "pnl": ["ItemType", "rule"],
                "oci": ["ItemType"],
                "date_columns": ["MaturityDate"],
            },
            "classifications": {
                "Book": "bank_projections.projections.book.BookRegistry",
            },
            "profitability_outlooks": ["Monthly", "Annual"],
        }

        config = AppConfig(**config_dict)

        assert config.labels.balance_sheet == ["ItemType"]
        assert config.profitability_outlooks == ["Monthly", "Annual"]
        assert len(config.classifications) == 1
        assert config.classifications[0].column_name == "Book"

    def test_app_config_get_classifications(self) -> None:
        """Test getting classifications dict with imported registries."""
        config_dict = {
            "labels": {
                "balance_sheet": ["ItemType"],
                "cashflow": ["ItemType"],
                "pnl": ["ItemType"],
                "oci": ["ItemType"],
                "date_columns": ["MaturityDate"],
            },
            "classifications": {
                "Book": "bank_projections.projections.book.BookRegistry",
                "HQLAClass": "bank_projections.financials.hqla_class.HQLARegistry",
            },
            "profitability_outlooks": ["Monthly"],
        }

        config = AppConfig(**config_dict)
        classifications = config.get_classifications()

        assert "Book" in classifications
        assert "HQLAClass" in classifications
        # Should be cached
        assert config.get_classifications() is classifications

    def test_app_config_label_columns(self) -> None:
        """Test label_columns combines all relevant columns."""
        config_dict = {
            "labels": {
                "balance_sheet": ["ItemType", "SubItemType"],
                "cashflow": ["ItemType"],
                "pnl": ["ItemType"],
                "oci": ["ItemType"],
                "date_columns": ["MaturityDate", "OriginationDate"],
            },
            "classifications": {
                "Book": "bank_projections.projections.book.BookRegistry",
            },
            "profitability_outlooks": ["Monthly"],
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
  balance_sheet: [CustomLabel]
  cashflow: [CustomCF]
  pnl: [CustomPnL]
  oci: [CustomOCI]
  date_columns: [CustomDate]

classifications:
  CustomClass: bank_projections.projections.book.BookRegistry

profitability_outlooks: [CustomOutlook]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = init_config(f.name)

            assert config.labels.balance_sheet == ["CustomLabel"]
            assert config.profitability_outlooks == ["CustomOutlook"]

        # Cleanup
        Path(f.name).unlink()
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
        assert "Book" in non_null
        assert "Nominal" in non_null
