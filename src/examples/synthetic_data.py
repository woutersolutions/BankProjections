import datetime
import os
import random

import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta

from bank_projections.financials.balance_sheet import BalanceSheet, Positions
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.financials.market_data import Curves, MarketRates
from bank_projections.projections.accrual_method import AccrualMethodRegistry
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.valuation_method import ValuationMethodRegistry
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.parsing import strip_identifier
from examples import EXAMPLE_FOLDER

SEED = 42

random.seed(SEED)


# TODO: Generate synthetic market data from csvs
def generate_synthetic_curves() -> Curves:
    return Curves(
        pd.DataFrame({"Name": "euribor", "Tenor": ["3m", "6m"], "Type": ["spot", "spot"], "Rate": [0.0285, 0.0305]})
    )


def generate_synthetic_positions(
    market_rates: MarketRates,
    book_value: float,
    number: int,
    balance_sheet_category: str,
    item_type: str,
    sub_item_type: str,
    accounting_method: str,
    accrual_method: str,
    redemption_type: str,
    coupon_frequency: str,
    current_date: datetime.date,
    coupon_type: str,
    curves: Curves = generate_synthetic_curves(),
    valuation_method: str | None = None,
    valuation_curve: str | None = None,
    currency: str = "EUR",
    hqla_class: str = "n/a",
    ifrs9_stage: str = "n/a",
    reference_rate: str | None = None,
    coverage_rate: float | tuple[float, float] | None = None,
    interest_rate: float | tuple[float, float] | None = None,
    undrawn_portion: float | tuple[float, float] | None = None,
    agio: float | tuple[float, float] | None = None,
    prepayment_rate: float | tuple[float, float] | None = None,
    ccf: float | tuple[float, float] | None = None,
    age: int | tuple[int, int] | None = None,
    maturity: int | tuple[int, int] | None = None,
    other_off_balance_weight: float | tuple[float, float] = 0.0,
    trea_weight: float | tuple[float, float] = 0.0,
    stable_funding_weight: float | tuple[float, float] = 0.0,
    stressed_outflow_weight: float | tuple[float, float] | None = None,
    encumbrance_weight: float | tuple[float, float] = 0.0,
    notional: float | tuple[float, float] | None = None,
) -> Positions:
    redemption_type = strip_identifier(redemption_type)
    coupon_frequency = strip_identifier(coupon_frequency)
    accounting_method = strip_identifier(accounting_method)
    reference_rate = strip_identifier(reference_rate)
    valuation_curve = strip_identifier(valuation_curve)
    valuation_method = "none" if valuation_method is None else strip_identifier(valuation_method)
    accrual_method = "none" if accrual_method is None else strip_identifier(accrual_method)

    # For notional-based instruments (like swaps), generate notionals separately
    if notional is not None:
        notionals = generate_values_from_input(number, notional)
        # Book values will be derived from notionals and market values
        book_values = None
    else:
        # Generate random book values that sum to the target book value
        if book_value == 0 or number == 1:
            book_values = [book_value] * number
        else:
            book_values = generate_random_numbers(
                number, 0.01, abs(book_value) * min(0.9, (100.0 / number)), abs(book_value) / number
            )
            # Scale so that the total book value matches exactly
            book_values = [value * book_value / sum(book_values) for value in book_values]
        notionals = None

    agios = generate_values_from_input(number, agio if agio is not None else 0.0)
    coverage_rates = generate_values_from_input(number, coverage_rate if coverage_rate is not None else 0.0)
    interest_rates = generate_values_from_input(number, interest_rate if interest_rate is not None else 0.0)
    undrawn_portions = generate_values_from_input(number, undrawn_portion if undrawn_portion is not None else 0.0)
    prepayment_rates = generate_values_from_input(number, prepayment_rate if prepayment_rate is not None else 0.0)
    ccf_values = generate_values_from_input(number, ccf if ccf is not None else 0.0)
    stressed_outflow_weights = generate_values_from_input(
        number, stressed_outflow_weight if stressed_outflow_weight is not None else 0.0
    )
    other_off_balance_weights = generate_values_from_input(number, other_off_balance_weight)
    trea_weights = generate_values_from_input(number, trea_weight)
    stable_funding_weights = generate_values_from_input(number, stable_funding_weight)
    encumbrance_weights = generate_values_from_input(number, encumbrance_weight)
    accrual_error_weights = generate_random_numbers(number, -0.01, 0.01, 0.0)

    coupon_type_stripped = strip_identifier(coupon_type)
    if coupon_type_stripped is None:
        raise ValueError(f"Invalid coupon_type: {coupon_type}")
    if coupon_type_stripped in CouponTypeRegistry.stripped_names():
        coupon_types = [coupon_type_stripped] * number
    elif coupon_type_stripped == "both":
        coupon_types = random.choices(["fixed", "floating"], weights=(0.6, 0.4), k=number)
    else:
        raise ValueError(f"Unknown coupon type: {coupon_type_stripped}")

    # Generate origination dates from age
    origination_dates: list[datetime.date | None]
    if age is None:
        origination_dates = [None] * number
    else:
        age_values = generate_int_values_from_input(number, age)
        origination_dates = [
            current_date - datetime.timedelta(days=age_val * 365) if age_val is not None else None
            for age_val in age_values
        ]

    # Generate maturity dates
    maturity_dates: list[datetime.date | None]
    match strip_identifier(redemption_type):
        case "perpetual":
            maturity_dates = [None] * number
        case "bullet" | "linear" | "annuity" | "notional":
            if maturity is None:
                raise ValueError(f"Maturity must be specified for redemption type: {redemption_type}")
            maturity_values = generate_int_values_from_input(number, maturity)
            maturity_dates = [
                current_date + relativedelta(years=mat_val) if mat_val is not None else None
                for mat_val in maturity_values
            ]
        case _:
            raise ValueError(f"Unknown redemption type: {redemption_type}")

    ifrs9_stage_stripped = strip_identifier(ifrs9_stage)
    if ifrs9_stage_stripped is None:
        raise ValueError(f"Invalid ifrs9_stage: {ifrs9_stage}")
    if ifrs9_stage_stripped == "mixed":
        ifrs9_stages = random.choices(["1", "2", "3", "poci"], weights=(0.9, 0.07, 0.02, 0.01), k=number)
    else:
        ifrs9_stages = [ifrs9_stage_stripped] * number

    clean_prices: list[float | None]
    if accounting_method == "amortizedcost":
        clean_prices = [None] * number
    elif valuation_method == "swap" and notionals is not None:
        # For swaps with notionals, derive CleanPrice from target book_value
        # book_value ≈ sum(Nominal * CleanPrice) = sum(notional * CleanPrice)
        # Assume equal CleanPrice across all swaps for simplicity
        total_notional = sum(notionals)
        avg_clean_price = book_value / total_notional if total_notional != 0 else 0.0
        clean_prices = [avg_clean_price] * number
    elif valuation_method == "swap":
        clean_prices = [0.0] * number
    else:
        clean_prices = [1.0] * number  # TODO: Do valuation to determine correct prices

    # Create polars dataframe with all the calculated fields
    df_dict = {
        "CoverageRate": coverage_rates,
        "CleanPrice": clean_prices,
        "AgioWeight": agios,
        "IFRS9Stage": ifrs9_stages,
        "InterestRate": interest_rates,
        "UndrawnPortion": undrawn_portions,
        "CouponType": coupon_types,
        "OriginationDate": origination_dates,
        "MaturityDate": maturity_dates,
        "PrepaymentRate": prepayment_rates,
        "CCF": ccf_values,
        "TREAWeight": trea_weights,
        "EncumberedWeight": encumbrance_weights,
        "StableFundingWeight": stable_funding_weights,
        "StressedOutflowWeight": stressed_outflow_weights,
        "OtherOffBalanceWeight": other_off_balance_weights,
        "AccruedInterestErrorWeight": accrual_error_weights,
    }

    # Add either book_values or notionals depending on instrument type
    if notionals is not None:
        df_dict["Notional"] = notionals
    else:
        df_dict["BookValue"] = book_values
        df_dict["Notional"] = 0.0

    df = (
        pl.DataFrame(
            df_dict,
            schema_overrides={
                "MaturityDate": pl.Date,
                "Notional": pl.Float64,
            },
        )
        .with_columns(
            ItemType=pl.lit(item_type),
            SubItemType=pl.lit(sub_item_type),
            Currency=pl.lit(strip_identifier(currency)),
            HQLAClass=pl.lit(strip_identifier(hqla_class)),
            RedemptionType=pl.lit(redemption_type),
            BalanceSheetCategory=pl.lit(balance_sheet_category),
            AccrualMethod=pl.lit(accrual_method),
            ValuationMethod=pl.lit(valuation_method),
            ValuationCurve=pl.lit(valuation_curve),
            ReferenceRate=pl.lit(reference_rate),
            CouponFrequency=pl.lit(coupon_frequency),
            AccountingMethod=pl.lit(accounting_method),
            Book=pl.lit("back"),
        )
        .with_columns(
            PreviousCouponDate=FrequencyRegistry.previous_coupon_date(
                current_date=current_date,
                anchor_date=pl.coalesce("MaturityDate", "OriginationDate", pl.lit(current_date)),
            ),
            NextCouponDate=FrequencyRegistry.next_coupon_date(
                current_date=current_date,
                anchor_date=pl.coalesce("MaturityDate", "OriginationDate", pl.lit(current_date)),
            ),
        )
        .with_columns(
            AccruedInterestWeight=AccrualMethodRegistry.get(accrual_method).calculate_current_accrued_interest(
                pl.lit(1.0),
                pl.col("InterestRate"),
                pl.col("PreviousCouponDate"),
                pl.col("NextCouponDate"),
                current_date,
            )
        )
    )

    # Handle nominal calculation differently for notional vs book_value instruments
    if notionals is not None:
        # For notional instruments (e.g., swaps), scale notionals to match target book value
        # Book value formula: sum(Nominal * CleanPrice + Nominal * AccruedInterestWeight)
        # Therefore: sum(Nominal * (CleanPrice + AccruedInterestWeight)) = book_value
        # Scale Notional proportionally: Nominal = Notional * scale_factor
        # where scale_factor ensures: sum(Notional * scale_factor * (CleanPrice + AccruedInterestWeight)) = book_value
        denominator = (pl.col("Notional") * (pl.col("CleanPrice") + pl.col("AccruedInterestWeight"))).sum()
        df = df.with_columns(
            Nominal=pl.lit(0.0),
            Notional=pl.when(denominator != 0.0).then(book_value / denominator * pl.col("Notional")).otherwise(0.0),
        )
    else:
        # For book_value instruments, derive Nominal from BookValue
        # For amortized cost: BookValue = Nominal + Agio + AccruedInterest + Impairment
        # For fair value: BookValue = Nominal * CleanPrice + AccruedInterest + Agio
        # Both can be expressed as: BookValue = Nominal * (1 + AgioWeight + AccruedInterestWeight - CoverageRate)
        df = df.with_columns(
            Nominal=pl.col("BookValue")
            / (1 + pl.col("AgioWeight") + pl.col("AccruedInterestWeight") - pl.col("CoverageRate"))
        ).drop("BookValue")

    df = df.with_columns(
        Impairment=-pl.col("Nominal") * pl.col("CoverageRate"),
        AccruedInterest=pl.col("Nominal") * pl.col("AccruedInterestWeight"),
        AccruedInterestError=pl.col("Nominal") * pl.col("AccruedInterestWeight") * pl.col("AccruedInterestErrorWeight"),
        Agio=pl.col("Nominal") * pl.col("AgioWeight"),
        Undrawn=pl.col("Nominal") * pl.col("UndrawnPortion"),
        ReferenceRate=pl.col("ReferenceRate").cast(pl.String),
        ValuationCurve=pl.col("ValuationCurve").cast(pl.String),
        CleanPrice=pl.col("CleanPrice").cast(pl.Float64),
    ).with_columns(
        FloatingRate=curves.floating_rate_expr(),
        Spread=pl.col("InterestRate") - curves.floating_rate_expr(),
        *[pl.lit(0.0).alias(column) for column in BalanceSheetMetrics.mutation_columns()],
    )

    # Perform valuation to initialize the t0 valuation error
    zero_rates = market_rates.curves.get_zero_rates()
    valuation_method_object = ValuationMethodRegistry.get(valuation_method)
    df = valuation_method_object.calculated_dirty_price(df, current_date, zero_rates, "CalculatedPrice")
    df = df.with_columns(
        valuation_method_object.valuation_error(
            pl.col("CalculatedPrice"), pl.col("CleanPrice") + pl.col("AccruedInterestWeight")
        ).alias("ValuationError"),
        (pl.col("CleanPrice") + pl.col("AccruedInterestWeight") / (pl.col("Nominal") + pl.col("Notional"))).alias(
            "DirtyPrice"
        ),
        (
            pl.when(pl.col("AccountingMethod") == "amortizedcost")
            .then(0.0)
            .otherwise(
                pl.col("CleanPrice") * (pl.col("Nominal") + pl.col("Notional"))
                - pl.col("Nominal")
                - pl.col("Impairment")
                - pl.col("AccruedInterest")
            )
        ).alias("FairValueAdjustment"),
    ).drop(
        [
            "AgioWeight",
            "AccruedInterestWeight",
            "AccruedInterestErrorWeight",
            "UndrawnPortion",
            "CoverageRate",
            "CalculatedPrice",
            "CleanPrice",
        ]
    )

    positions = Positions(df)

    positions.validate()

    # Always validate book_value matches target (works for both notional and non-notional instruments)
    generated_bv = positions.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
    assert abs(generated_bv - book_value) < 1e-2, (
        f"Generated book value {generated_bv} not equal to target {book_value}"
    )

    return positions


def generate_random_numbers(number: int, minimum: float, maximum: float, mean: float) -> list[float]:
    # Use beta distribution to generate numbers
    if mean <= minimum:
        raise ValueError(f"Mean {mean} must be greater than minimum {minimum}")
    if mean >= maximum:
        raise ValueError(f"Mean {mean} must be less than maximum {maximum}")

    alpha = ((mean - minimum) / (maximum - minimum)) * 5
    beta = ((maximum - mean) / (maximum - minimum)) * 5

    # Seed NumPy from Python's RNG to keep reproducibility with `random.seed(...)`
    rng = np.random.default_rng(random.getrandbits(64))
    arr = rng.beta(alpha, beta, size=number) * (maximum - minimum) + minimum
    return arr.tolist()


def generate_values_from_input(number: int, value: float | tuple[float, float]) -> list[float]:
    """
    Generate a list of values from either a single value or a range.

    - If single value, return list of that value
    - If tuple (min, max), generate random values in that range
    """
    if isinstance(value, tuple):
        minimum, maximum = value
        mean = (minimum + maximum) / 2
        if minimum == maximum:
            return [minimum] * number
        return generate_random_numbers(number, minimum, maximum, mean)
    else:
        return [float(value)] * number


def generate_int_values_from_input(
    number: int, value: int | float | tuple[int | float, int | float] | None
) -> list[int | None]:
    """
    Generate a list of integer values from either a single value or a range.

    - If None, return list of None values
    - If single value, return list of that value (converted to int)
    - If tuple (min, max), generate random integer values in that range
    """
    if value is None:
        return [None] * number
    elif isinstance(value, tuple):
        minimum, maximum = int(value[0]), int(value[1])
        return [random.randint(minimum, maximum) for _ in range(number)]
    else:
        return [int(value)] * number


def create_synthetic_balance_sheet(
    current_date: datetime.date,
    scenario: Scenario,
    config_path: str | None = os.path.join(EXAMPLE_FOLDER, "example_bs.csv"),
    config_table: pl.DataFrame | None = None,
) -> BalanceSheet:
    # Iterate over synthetic_data.csv using polars to create each of the items

    if config_table is None:
        if config_path is None:
            raise ValueError("Either config_table or config_path must be provided")
        config_table = pl.read_csv(config_path)

    curves = generate_synthetic_curves()

    market_rates = scenario.market_data.get_market_rates(current_date)

    # List of numeric columns that should be parsed with read_range
    numeric_columns = {
        "coverage_rate",
        "interest_rate",
        "undrawn_portion",
        "agio",
        "prepayment_rate",
        "ccf",
        "age",
        "maturity",
        "other_off_balance_weight",
        "trea_weight",
        "stable_funding_weight",
        "stressed_outflow_weight",
        "encumbrance_weight",
        "notional",
    }

    positions = []
    for row in config_table.iter_rows(named=True):
        position_input = {name: read_range(value) if name in numeric_columns else value for name, value in row.items()}
        if row["number"] > 0:
            positions.append(
                generate_synthetic_positions(
                    market_rates=market_rates, current_date=current_date, curves=curves, **position_input
                )
            )

    combined_positions = Positions.combine(*positions)

    bs = BalanceSheet(combined_positions._data, current_date)

    return bs


def read_range(value: str | float | int) -> float | tuple[float, float] | None:
    """
    Parse a value that can be either:
    - None or empty string -> None
    - Already a number (from CSV parsing) -> return as is
    - Single number string "0.5" -> 0.5
    - Range string "(0.1, 0.5)" -> (0.1, 0.5)
    """
    # Handle None
    if value is None:
        return None

    # If already a number (int or float), return it
    if isinstance(value, int | float):
        return float(value)

    # Handle string values
    if isinstance(value, str):
        value = value.strip()
        if len(value) == 0:
            return None

        # Check if it's a range (has parentheses and comma)
        if value.startswith("(") and value.endswith(")") and "," in value:
            parts = value.strip("()").split(",")
            if len(parts) == 2:
                return tuple(float(x.strip()) for x in parts)
            else:
                raise ValueError(f"Invalid range format: {value}. Expected format: (min, max)")
        else:
            # Single number string
            try:
                return float(value)
            except ValueError as err:
                raise ValueError(f"Invalid numeric value: {value}. Expected a number or range (min, max)") from err

    raise ValueError(f"Unexpected value type: {type(value)}")


def create_single_asset_balance_sheet(
    current_date: datetime.date,
    scenario: Scenario,
    book_value: float,
    accounting_method: str,
    redemption_type: str,
    coupon_frequency: str,
    coupon_type: str,
    item_type: str = "Loan",
    sub_item_type: str = "Loan",
    ifrs9_stage: str | None = None,
    coverage_rate: float | tuple[float, float] | None = None,
    interest_rate: float | tuple[float, float] | None = None,
    undrawn_portion: float | tuple[float, float] | None = None,
    agio: float | tuple[float, float] | None = None,
    prepayment_rate: float | tuple[float, float] | None = None,
    age: int | tuple[int, int] | None = None,
    maturity: int | tuple[int, int] | None = None,
    other_off_balance_weight: float | tuple[float, float] | None = None,
    trea_weight: float | tuple[float, float] | None = None,
    reference_rate: str | None = None,
    notional: float | tuple[float, float] | None = None,
    stable_funding_weight: float | tuple[float, float] | None = None,
    encumbrance_weight: float | tuple[float, float] | None = None,
    hqla_class: str | None = None,
    stressed_outflow_weight: float | tuple[float, float] | None = None,
    ccf: float | tuple[float, float] | None = None,
    valuation_method: str | None = None,
    valuation_curve: str | None = None,
    config_table: pl.DataFrame | None = None,
) -> BalanceSheet:
    """
    Create a synthetic balance sheet with a specified asset added to the default configuration.

    This is a demo/UI function that adds a new asset with the specified parameters to the
    default balance sheet configuration. Useful for demonstrating runoff behavior of different
    asset types in a realistic balance sheet environment.

    Parameters
    ----------
    current_date : datetime.date
        Starting date for the balance sheet
    scenario : Scenario
        Scenario configuration with market data and curves
    item_type : str
        Type of asset (e.g., "Loans", "Debt securities")
    sub_item_type : str
        Subtype of asset (e.g., "Mortgages", "SME loans")
    book_value : float
        Total book value for the asset
    number : int
        Number of positions to generate
    accounting_method : str
        Accounting method (e.g., "amortized cost", "fair value through oci")
    redemption_type : str
        Redemption type (e.g., "annuity", "bullet", "perpetual")
    coupon_frequency : str
        Coupon payment frequency (e.g., "Monthly", "Quarterly")
    coupon_type : str
        Type of coupon (e.g., "fixed", "floating", "both")
    config_table : pl.DataFrame | None
        Optional config table to use instead of loading default

    Returns
    -------
    BalanceSheet
        Synthetic balance sheet with the specified asset added
    """
    # Load default config if not provided
    if config_table is None:
        config_path = os.path.join(EXAMPLE_FOLDER, "example_bs.csv")
        config_table = pl.read_csv(config_path)

    # Use the full default config with the new asset added
    # This ensures balance and provides a realistic demo environment
    modified_config = config_table

    # Helper to format values for CSV string columns
    def format_string_value(value: float | tuple[float, float] | int | tuple[int, int] | str | None) -> str:
        if value is None:
            return ""
        elif isinstance(value, tuple):
            return f"({value[0]}, {value[1]})"
        else:
            return str(value)

    # Determine valuation_method if not provided
    if valuation_method is None or valuation_method == "":
        # Default based on accounting method
        if "amortized cost" in accounting_method.lower():
            valuation_method = "amortizedcost"
        elif "fair value" in accounting_method.lower():
            # For fair value, a specific valuation method should be provided
            # Default to none if not specified
            valuation_method = "none"
        else:
            valuation_method = "none"

    # Default enum fields to valid values if not provided
    if hqla_class is None or hqla_class == "":
        hqla_class = "non-HQLA"  # Default for non-HQLA assets

    if ifrs9_stage is None or ifrs9_stage == "":
        ifrs9_stage = "n/a"  # Default when IFRS9 stage is not applicable

    # Helper to format float values
    def format_float_value(value: float | tuple[float, float] | None) -> float:
        if value is None:
            return 0.0  # Default to 0.0 for optional weight parameters
        elif isinstance(value, tuple):
            # For weight columns, use the midpoint of the range
            return (value[0] + value[1]) / 2.0
        else:
            return float(value)

    # Create new row for the single asset - match CSV schema exactly
    new_row = pl.DataFrame(
        {
            "balance_sheet_category": pl.Series(["assets"], dtype=pl.String),
            "item_type": pl.Series([item_type], dtype=pl.String),
            "sub_item_type": pl.Series([sub_item_type], dtype=pl.String),
            "currency": pl.Series(["eur"], dtype=pl.String),
            "book_value": pl.Series([int(book_value)], dtype=pl.Int64),
            "number": pl.Series([1], dtype=pl.Int64),
            "accounting_method": pl.Series([accounting_method], dtype=pl.String),
            "valuation_method": pl.Series([format_string_value(valuation_method)], dtype=pl.String),
            "valuation_curve": pl.Series([format_string_value(valuation_curve)], dtype=pl.String),
            "redemption_type": pl.Series([redemption_type], dtype=pl.String),
            "coupon_frequency": pl.Series([coupon_frequency], dtype=pl.String),
            "coupon_type": pl.Series([coupon_type], dtype=pl.String),
            "ifrs9_stage": pl.Series([format_string_value(ifrs9_stage)], dtype=pl.String),
            "coverage_rate": pl.Series([format_string_value(coverage_rate)], dtype=pl.String),
            "interest_rate": pl.Series([format_string_value(interest_rate)], dtype=pl.String),
            "undrawn_portion": pl.Series([format_string_value(undrawn_portion)], dtype=pl.String),
            "agio": pl.Series([format_string_value(agio)], dtype=pl.String),
            "prepayment_rate": pl.Series([format_float_value(prepayment_rate)], dtype=pl.Float64),
            "age": pl.Series([format_string_value(age)], dtype=pl.String),
            "maturity": pl.Series([format_string_value(maturity)], dtype=pl.String),
            "other_off_balance_weight": pl.Series([format_float_value(other_off_balance_weight)], dtype=pl.Float64),
            "trea_weight": pl.Series([format_float_value(trea_weight)], dtype=pl.Float64),
            "reference_rate": pl.Series([format_string_value(reference_rate)], dtype=pl.String),
            "notional": pl.Series([format_string_value(notional)], dtype=pl.String),
            "stable_funding_weight": pl.Series([format_float_value(stable_funding_weight)], dtype=pl.Float64),
            "encumbrance_weight": pl.Series([format_float_value(encumbrance_weight)], dtype=pl.Float64),
            "hqla_class": pl.Series([format_string_value(hqla_class)], dtype=pl.String),
            "stressed_outflow_weight": pl.Series([format_float_value(stressed_outflow_weight)], dtype=pl.Float64),
            "ccf": pl.Series([format_float_value(ccf)], dtype=pl.Float64),
        }
    )

    # Add the new asset row
    modified_config = pl.concat([modified_config, new_row], how="diagonal")

    # Balance the sheet by:
    # 1. Keep only the new asset with its book_value
    # 2. Keep Cash with book value to match the asset (required for BalanceSheet initialization)
    # 3. Zero out book_value for all other items (number stays > 0 to maintain schema compatibility)
    # 4. Set Retained earnings to balance the sheet
    # Note: Items with book_value=0 but number>0 create zero-nominal positions (minimal impact)
    modified_config = modified_config.with_columns(
        pl.when(
            # Keep the new asset with its book_value
            (pl.col("balance_sheet_category") == "assets") & (pl.col("sub_item_type") == sub_item_type)
        )
        .then(pl.col("book_value"))
        .when(
            # Keep Cash with book_value matching the asset (required for BalanceSheet initialization)
            (pl.col("balance_sheet_category") == "assets") & (pl.col("item_type") == "Cash")
        )
        .then(pl.lit(int(book_value)).cast(pl.Int64))
        .when(
            # Zero out all other assets
            pl.col("balance_sheet_category") == "assets"
        )
        .then(pl.lit(0).cast(pl.Int64))
        .when(
            # Zero out all liabilities
            pl.col("balance_sheet_category") == "liabilities"
        )
        .then(pl.lit(0).cast(pl.Int64))
        .when(
            # Set Retained earnings to (asset book_value + cash) to balance the sheet
            (pl.col("balance_sheet_category") == "equity") & (pl.col("sub_item_type") == "Retained earnings")
        )
        .then(pl.lit(2 * int(book_value)).cast(pl.Int64))
        .when(
            # Zero out all other equity items
            pl.col("balance_sheet_category") == "equity"
        )
        .then(pl.lit(0).cast(pl.Int64))
        # Zero out everything else (derivatives, etc.)
        .otherwise(pl.lit(0).cast(pl.Int64))
        .alias("book_value")
    )

    # Ensure all rows have valid valuation_method, hqla_class, and ifrs9_stage
    modified_config = modified_config.with_columns(
        # Default valuation_method based on accounting_method if empty
        pl.when((pl.col("valuation_method") == "") | pl.col("valuation_method").is_null())
        .then(
            pl.when(pl.col("accounting_method").str.contains("(?i)amortized cost"))
            .then(pl.lit("amortizedcost"))
            .otherwise(pl.lit("none"))
        )
        .otherwise(pl.col("valuation_method"))
        .alias("valuation_method"),
        # Default hqla_class if empty
        pl.when((pl.col("hqla_class") == "") | pl.col("hqla_class").is_null())
        .then(pl.lit("non-HQLA"))
        .otherwise(pl.col("hqla_class"))
        .alias("hqla_class"),
        # Default ifrs9_stage if empty
        pl.when((pl.col("ifrs9_stage") == "") | pl.col("ifrs9_stage").is_null())
        .then(pl.lit("n/a"))
        .otherwise(pl.col("ifrs9_stage"))
        .alias("ifrs9_stage"),
    )

    # Generate the balance sheet using the standard function
    bs = create_synthetic_balance_sheet(current_date, scenario, config_table=modified_config)

    # Filter out zero-nominal positions from Assets and Liabilities
    # (keep all Equity positions for balance sheet integrity)
    bs._data = bs._data.filter(
        (pl.col("Nominal").abs() > 0.00001) | ~pl.col("BalanceSheetCategory").is_in(["assets", "liabilities"])
    )

    return bs
