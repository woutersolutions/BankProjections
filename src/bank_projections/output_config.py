from pydantic import BaseModel


class AggregationConfig(BaseModel):
    balance_sheet: list[str] | None = None
    pnl: list[str] | None = None
    cashflow: list[str] | None = None
    oci: list[str] | None = None


class OutputConfig(BaseModel):
    output_folder: str
    output_file: str
    aggregation: AggregationConfig
