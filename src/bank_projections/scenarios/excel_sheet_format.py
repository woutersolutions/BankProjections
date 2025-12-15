"""Excel template loader with automatic template detection."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bank_projections.utils.base_registry import BaseRegistry
from bank_projections.utils.parsing import strip_identifier


@dataclass
class ExcelInput:
    """Unified container for Excel input data from any template type."""

    content: pd.DataFrame = field(default_factory=pd.DataFrame)
    col_headers: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_headers: pd.DataFrame = field(default_factory=pd.DataFrame)
    general_tags: dict[str, Any] = field(default_factory=dict)
    template_name: str = ""

    def __post_init__(self):
        if self.row_headers.empty and not self.content.empty:
            self.row_headers = pd.DataFrame(index=self.content.index)
        self.template_name = strip_identifier(self.template_name)


class TemplateType(ABC):
    """Abstract base class for Excel template types."""

    @classmethod
    @abstractmethod
    def matches(cls, df_raw: pd.DataFrame) -> bool:
        """Check if this template type matches the given raw DataFrame."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str, sheet_name: str, df_raw: pd.DataFrame, template_name: str) -> ExcelInput:
        """Load the Excel sheet into an ExcelInput."""
        pass


class TemplateTypeRegistry(BaseRegistry[type[TemplateType]]):
    """Registry for Excel template types."""

    @classmethod
    def load_paths(cls, paths: list[str]) -> list[ExcelInput]:
        """
        Load multiple paths and return combined list of ExcelInput objects.

        Args:
            paths: List of file or folder paths to load.

        Returns:
            List of ExcelInput objects from all paths.
        """
        results = []
        for path in paths:
            results.extend(cls.load_path(path))
        return results

    @classmethod
    def load_path(cls, path: str) -> list[ExcelInput]:
        """
        Load a single path (file or folder).

        Args:
            path: Path to a file or folder.

        Returns:
            List of ExcelInput objects.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        if os.path.isdir(path):
            return cls.load_folder(path)

        _name, extension = os.path.splitext(path)

        match extension:
            case ".xlsx" | ".xls":
                return cls.load_excel(path)
            case _:
                raise ValueError(f"Unsupported file type: {extension}")

    @classmethod
    def load_folder(cls, folder_path: str) -> list[ExcelInput]:
        """
        Load all Excel files in a folder.

        Args:
            folder_path: Path to the folder.

        Returns:
            List of ExcelInput objects from all files in the folder.
        """
        results = []
        for file_name in os.listdir(folder_path):
            # Ignore temporary files
            if file_name.startswith("~$"):
                continue

            results.extend(cls.load_path(os.path.join(folder_path, file_name)))
        return results

    @classmethod
    def load_excel(cls, file_path: str) -> list[ExcelInput]:
        """
        Load all sheets from an Excel file.

        Args:
            file_path: Path to the Excel file.

        Returns:
            List of ExcelInput objects from all sheets.
        """
        xls = pd.ExcelFile(file_path)
        results = []
        for sheet_name in xls.sheet_names:
            result = cls.load_excel_sheet(file_path, sheet_name)
            results.append(result)
        return results

    @classmethod
    def detect_template_type(cls, df_raw: pd.DataFrame) -> type[TemplateType]:
        """Detect which template type applies to the given raw DataFrame."""
        for template_type in cls.values():
            if template_type.matches(df_raw):
                return template_type
        raise ValueError("No matching template type found")

    @classmethod
    def load_excel_sheet(cls, file_path: str, sheet_name: str) -> ExcelInput:
        """
        Load an Excel sheet by auto-detecting the template type.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Name of the sheet to load.

        Returns:
            ExcelInput containing the parsed data.
        """
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Validate template header
        if strip_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")
        template_name = strip_identifier(str(df_raw.iloc[0, 1]))

        template_type = cls.detect_template_type(df_raw)
        return template_type.load(file_path, sheet_name, df_raw, template_name)

    @classmethod
    def load_excel_sheet_with_type(cls, file_path: str, sheet_name: str, type_name: str) -> ExcelInput:
        """
        Load an Excel sheet using a specific template type.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Name of the sheet to load.
            type_name: The registered name of the template type to use.

        Returns:
            ExcelInput containing the parsed data.
        """
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Validate template header
        if strip_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")
        template_name = strip_identifier(str(df_raw.iloc[0, 1]))

        template_type = cls.get(type_name)
        return template_type.load(file_path, sheet_name, df_raw, template_name)


class MultiHeaderTemplate(TemplateType):
    """Template with multi-row column headers and row headers, marked by '*'."""

    @classmethod
    def matches(cls, df_raw: pd.DataFrame) -> bool:
        return df_raw.map(lambda x: isinstance(x, str) and "*" in x).any().any()

    @classmethod
    def load(cls, file_path: str, sheet_name: str, df_raw: pd.DataFrame, template_name: str) -> ExcelInput:
        # Find cell with '*' marker
        star_mask = df_raw.map(lambda x: isinstance(x, str) and "*" in x)
        star_row, star_col = star_mask.stack()[star_mask.stack()].index[0]

        # Find the first row with non-empty cells from the third column onwards
        col_header_start_row = df_raw.iloc[:, 2:].apply(lambda row: row.notna().any(), axis=1).idxmax()
        assert col_header_start_row <= star_row

        # Extract column headers (rows from col_header_start to star_row, columns from star_col onwards)
        col_headers = (
            df_raw.iloc[col_header_start_row : (star_row + 1), star_col:].set_index(star_col).T.reset_index(drop=True)
        )
        col_headers.columns = [str(col).split("*")[-1] for col in col_headers.columns]

        # Extract row headers (rows after star_row, columns up to and including star_col)
        row_headers = df_raw.iloc[(star_row + 1) :, : (star_col + 1)].reset_index(drop=True)
        row_headers.columns = df_raw.iloc[star_row, : (star_col + 1)].values
        row_headers = row_headers.rename(columns={row_headers.columns[-1]: str(row_headers.columns[-1]).split("*")[0]})

        # Read the content table
        content = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=star_row + 1,
            usecols=range(star_col + 1, df_raw.shape[1]),
        )

        # Read general tags (key-value pairs in columns A:B between row 1 and header start)
        general_tags = {}
        for idx in range(1, col_header_start_row):
            key = str(df_raw.iloc[idx, 0]).strip()
            value = df_raw.iloc[idx, 1]
            if key and pd.notna(value):
                general_tags[key] = value

        return ExcelInput(
            content=content,
            col_headers=col_headers,
            row_headers=row_headers,
            general_tags=general_tags,
            template_name=template_name,
        )


class KeyValueTemplate(TemplateType):
    """Template with only key-value pairs in columns A:B."""

    @classmethod
    def matches(cls, df_raw: pd.DataFrame) -> bool:
        if df_raw.shape[1] <= 2:
            return True
        cols_beyond_ab = df_raw.iloc[:, 2:]
        return cols_beyond_ab.isna().all().all() or cols_beyond_ab.empty

    @classmethod
    def load(cls, file_path: str, sheet_name: str, df_raw: pd.DataFrame, template_name: str) -> ExcelInput:
        # Read key-value pairs from rows below the template header
        general_tags = {}
        for idx in range(1, len(df_raw)):
            key = df_raw.iloc[idx, 0]
            value = df_raw.iloc[idx, 1]
            if pd.notna(key) and pd.notna(value):
                general_tags[str(key).strip()] = value

        return ExcelInput(
            general_tags=general_tags,
            template_name=template_name,
        )


class OneHeaderTemplate(TemplateType):
    """Template with a single header row."""

    @classmethod
    def matches(cls, df_raw: pd.DataFrame) -> bool:
        # OneHeader is the fallback - matches if others don't
        return True

    @classmethod
    def load(cls, file_path: str, sheet_name: str, df_raw: pd.DataFrame, template_name: str) -> ExcelInput:
        # Find the first row with non-empty cells from the third column onwards
        header_start_row = df_raw.iloc[:, 2:].apply(lambda row: row.notna().any(), axis=1).idxmax()

        # Read the content with headers
        content = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header_start_row,
        )

        # Create col_headers from the column names
        col_headers = pd.DataFrame({"column": content.columns})

        # Read general tags
        general_tags = {}
        for idx in range(1, header_start_row):
            key = str(df_raw.iloc[idx, 0]).strip()
            value = df_raw.iloc[idx, 1]
            if key and pd.notna(value):
                general_tags[key] = value

        return ExcelInput(
            content=content,
            col_headers=col_headers,
            general_tags=general_tags,
            template_name=template_name,
        )


# Register template types in order of precedence
TemplateTypeRegistry.register("multi_header", MultiHeaderTemplate)
TemplateTypeRegistry.register("key_value", KeyValueTemplate)
TemplateTypeRegistry.register("one_header", OneHeaderTemplate)
