"""Application configuration with singleton pattern.

This module provides global application configuration that is loaded from a YAML file
and dictionary CSV, accessible throughout the codebase via the `get_config()` function.

The configuration auto-initializes from a default path when first accessed, making it
seamless to use throughout the codebase without explicit initialization.
"""

import csv
import importlib
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, field_validator

from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.utils.base_registry import BaseRegistry

# Default config paths relative to this module
DEFAULT_CONFIG_PATH = Path(__file__).parent / "app_config.yaml"
DEFAULT_DICTIONARY_PATH = Path(__file__).parent / "dictionary.csv"

DataType = Literal["money", "percentage", "float", "string", "date", "enum"]
KeywordType = Literal[
    "StoredAmount", "StoredWeight", "MutationAmount", "DerivedMetric", "Label", "DateColumn", "Classification"
]


class DictionaryEntry(BaseModel):
    """A single entry from the dictionary CSV."""

    keyword: str
    data_type: DataType
    required: bool
    keyword_type: KeywordType
    registry: str | None = None
    description: str

    @field_validator("required", mode="before")
    @classmethod
    def parse_required(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("yes", "true", "1")
        return bool(v)

    @field_validator("registry", mode="before")
    @classmethod
    def parse_registry(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        return str(v)


class ClassificationConfig(BaseModel):
    """Configuration for a single classification column with its registry."""

    column_name: str
    registry_import: str

    def get_registry(self) -> type[BaseRegistry[Any]]:
        """Dynamically import and return the registry class."""
        module_path, class_name = self.registry_import.rsplit(".", 1)
        module = importlib.import_module("bank_projections." + module_path)
        registry: type[BaseRegistry[Any]] = getattr(module, class_name)
        return registry


class LabelsConfig(BaseModel):
    """Configuration for column labels used in various data structures."""

    cashflow: list[str]
    pnl: list[str]
    oci: list[str]


class AppConfig(BaseModel):
    """Main application configuration loaded from YAML and dictionary CSV."""

    labels: LabelsConfig
    profitability_outlooks: list[str]
    dictionary: list[DictionaryEntry] = []

    _registry_cache: dict[str, type[BaseRegistry[Any]]] | None = None

    def get_dictionary_entries(self, keyword_type: KeywordType | None = None) -> list[DictionaryEntry]:
        """Get dictionary entries, optionally filtered by metric type."""
        if keyword_type is None:
            return self.dictionary
        return [e for e in self.dictionary if e.keyword_type == keyword_type]

    def get_dictionary_entry(self, keyword: str) -> DictionaryEntry | None:
        """Get a single dictionary entry by keyword."""
        for entry in self.dictionary:
            if entry.keyword == keyword:
                return entry
        return None

    def get_classifications(self) -> dict[str, type[BaseRegistry[Any]]]:
        """Get the classifications mapping, caching the imported registries."""
        if self._registry_cache is None:
            classifications = {}
            for entry in self.get_dictionary_entries("Classification"):
                if entry.registry:
                    config = ClassificationConfig(column_name=entry.keyword, registry_import=entry.registry)
                    classifications[entry.keyword] = config.get_registry()
            object.__setattr__(self, "_registry_cache", classifications)
        return self._registry_cache  # type: ignore[return-value]

    def balance_sheet_labels(self) -> list[str]:
        """Get balance sheet label columns from dictionary."""
        return [e.keyword for e in self.get_dictionary_entries("Label")]

    def cashflow_labels(self) -> list[str]:
        return self.labels.cashflow

    def pnl_labels(self) -> list[str]:
        return self.labels.pnl

    def oci_labels(self) -> list[str]:
        return self.labels.oci

    def date_columns(self) -> list[str]:
        """Get date columns from dictionary."""
        return [e.keyword for e in self.get_dictionary_entries("DateColumn")]

    def label_columns(self) -> list[str]:
        """Get all label columns (balance sheet labels + date columns + classification keys)."""
        return self.balance_sheet_labels() + self.date_columns() + list(self.get_classifications().keys())

    def required_columns(self) -> list[str]:
        """Get all required columns for a balance sheet."""

        return (
            self.balance_sheet_labels()
            + list(self.get_classifications().keys())
            + BalanceSheetMetrics.stored_columns()
            + self.date_columns()
        )

    def non_null_columns(self) -> list[str]:
        """Get columns that should not contain null values (required columns from dictionary)."""
        return [e.keyword for e in self.dictionary if e.required]

    def cast_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast classification columns to their enum types."""
        return df.with_columns(
            [
                pl.col(name).cast(pl.Enum(registry.stripped_names()))
                for name, registry in self.get_classifications().items()
            ]
        )


# Singleton instance
_config_instance: AppConfig | None = None
_config_path: Path | None = None


def load_dictionary(dictionary_path: Path) -> list[DictionaryEntry]:
    """Load dictionary entries from a CSV file."""
    entries = []
    with open(dictionary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(DictionaryEntry.model_validate(row))
    return entries


def init_config(
    config_path: str | Path | None = None,
    dictionary_path: str | Path | None = None,
) -> AppConfig:
    """Initialize the global configuration from a YAML file and dictionary CSV.

    Parameters
    ----------
    config_path : str | Path | None
        Path to the YAML configuration file. If None, uses the default
        config file located alongside this module.
    dictionary_path : str | Path | None
        Path to the dictionary CSV file. If None, uses the default
        dictionary file located alongside this module.

    Returns
    -------
    AppConfig
        The initialized configuration instance
    """
    global _config_instance, _config_path

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if dictionary_path is None:
        dictionary_path = DEFAULT_DICTIONARY_PATH

    config_path = Path(config_path)
    dictionary_path = Path(dictionary_path)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict["dictionary"] = load_dictionary(dictionary_path)

    _config_instance = AppConfig(**config_dict)
    _config_path = config_path
    return _config_instance


def get_config() -> AppConfig:
    """Get the global configuration instance.

    If the configuration has not been explicitly initialized, this will
    auto-initialize from the default configuration file.

    Returns
    -------
    AppConfig
        The global configuration instance
    """
    global _config_instance
    if _config_instance is None:
        init_config()
    assert _config_instance is not None
    return _config_instance


def is_config_initialized() -> bool:
    """Check if the configuration has been initialized."""
    return _config_instance is not None


def reset_config() -> None:
    """Reset the configuration singleton. Mainly useful for testing."""
    global _config_instance, _config_path
    _config_instance = None
    _config_path = None


class _ConfigProxy:
    """Proxy class that delegates all attribute access to the singleton AppConfig instance."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_config(), name)


# Convenience proxy - allows Config.method() syntax without calling Config() first
Config = _ConfigProxy()
