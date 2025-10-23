import datetime
import os
import random

import numpy as np
import pandas as pd
import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, Positions
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetrics
from bank_projections.financials.market_data import Curves, MarketRates
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry, interest_accrual
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
    balance_sheet_side: str,
    item_type: str,
    sub_item_type: str,
    accounting_method: str,
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
    coverage_rate_range: tuple[float, float] | None = None,
    interest_rate_range: tuple[float, float] | None = None,
    undrawn_portion_range: tuple[float, float] | None = None,
    agio_range: tuple[float, float] | None = None,
    prepayment_rate: float | None = 0.0,
    ccf: float | None = 0.0,
    minimum_age: int | None = None,
    maximum_age: int | None = None,
    minimum_maturity: int | None = None,
    maximum_maturity: int | None = None,
    accumulating: bool | None = False,
    other_off_balance_weight: float = 0.0,
    trea_weight: float = 0.0,
    stable_funding_weight: float = 0.0,
    stressed_outflow_weight: float | None = None,
    encumbrance_weight: float = 0.0,
    notional_range: tuple[float, float] | None = None,
) -> Positions:
    redemption_type = strip_identifier(redemption_type)
    coupon_frequency = strip_identifier(coupon_frequency)
    accounting_method = strip_identifier(accounting_method)
    reference_rate = strip_identifier(reference_rate)
    valuation_curve = strip_identifier(valuation_curve)
    valuation_method = "none" if valuation_method is None else strip_identifier(valuation_method)

    # For notional-based instruments (like swaps), generate notionals separately
    if notional_range is not None:
        notionals = generate_random_numbers(
            number, notional_range[0], notional_range[1], (notional_range[0] + notional_range[1]) / 2
        )
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

    if agio_range is None:
        agios = [0.0] * number
    else:
        agios = generate_random_numbers(number, agio_range[0], agio_range[1], (agio_range[0] + agio_range[1]) / 2)

    if prepayment_rate is None:
        prepayment_rate = 0.0
    if ccf is None:
        ccf = 0.0
    if stressed_outflow_weight is None:
        stressed_outflow_weight = 0.0

    if coverage_rate_range is None:
        coverage_rates = [0.0] * number
    else:
        coverage_rates = generate_random_numbers(
            number,
            coverage_rate_range[0],
            coverage_rate_range[1],
            (coverage_rate_range[0] + coverage_rate_range[1]) / 2,
        )
    if interest_rate_range is None:
        interest_rates = [0.0] * number
    else:
        interest_rates = generate_random_numbers(
            number,
            interest_rate_range[0],
            interest_rate_range[1],
            (interest_rate_range[0] + interest_rate_range[1]) / 2,
        )

    if undrawn_portion_range is None:
        undrawn_portions = [0.0] * number
    else:
        undrawn_portions = generate_random_numbers(
            number,
            undrawn_portion_range[0],
            undrawn_portion_range[1],
            (undrawn_portion_range[0] + undrawn_portion_range[1]) / 2,
        )

    coupon_type = strip_identifier(coupon_type)
    if coupon_type in CouponTypeRegistry.stripped_names():
        coupon_types = [strip_identifier(coupon_type)] * number
    elif coupon_type == "both":
        coupon_types = random.choices(["fixed", "floating"], weights=(0.6, 0.4), k=number)
    else:
        raise ValueError(f"Unknown coupon type: {coupon_type}")

    if minimum_age is None and maximum_age is None:
        origination_dates = [None] * number
    elif maximum_age is None:
        raise ValueError("If minimum_age is set, maximum_age must also be set")
    else:
        if minimum_age is None:
            minimum_age = 0
        origination_dates = [
            current_date - datetime.timedelta(days=random.randint(minimum_age * 365, maximum_age * 365))
            for _ in range(number)
        ]

    match strip_identifier(redemption_type):
        case "perpetual":
            maturity_dates = [None] * number
        case "bullet" | "linear" | "annuity" | "notional":
            # Generate uniform random based on current date and min/max maturity in years,
            # by adding a random number of days
            maturity_dates = [
                current_date + datetime.timedelta(days=random.randint(minimum_maturity * 365, maximum_maturity * 365))
                for _ in range(number)
            ]
        case _:
            raise ValueError(f"Unknown redemption type: {redemption_type}")

    ifrs9_stage = strip_identifier(ifrs9_stage)
    if ifrs9_stage == "mixed":
        ifrs9_stages = random.choices(["1", "2", "3", "poci"], weights=(0.9, 0.07, 0.02, 0.01), k=number)
    else:
        ifrs9_stages = [ifrs9_stage] * number

    if accounting_method == "amortizedcost":
        clean_prices = [None] * number
    elif valuation_method == "swap" and notionals is not None:
        # For swaps with notionals, derive CleanPrice from target book_value
        # book_value ≈ sum(Quantity * CleanPrice) = sum(notional * CleanPrice)
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
    }

    # Add either book_values or notionals depending on instrument type
    if notionals is not None:
        df_dict["Notional"] = notionals
    else:
        df_dict["BookValue"] = book_values

    df = (
        pl.DataFrame(
            df_dict,
            schema_overrides={
                "MaturityDate": pl.Date,
            },
        )
        .with_columns(
            ItemType=pl.lit(item_type),
            SubItemType=pl.lit(sub_item_type),
            Currency=pl.lit(strip_identifier(currency)),
            HQLAClass=pl.lit(strip_identifier(hqla_class)),
            PrepaymentRate=pl.lit(prepayment_rate),
            CCF=pl.lit(ccf),
            IsAccumulating=pl.lit(accumulating),
            RedemptionType=pl.lit(redemption_type),
            BalanceSheetSide=pl.lit(balance_sheet_side),
            TREAWeight=pl.lit(trea_weight),
            EncumberedWeight=pl.lit(encumbrance_weight),
            StableFundingWeight=pl.lit(stable_funding_weight),
            StressedOutflowWeight=pl.lit(stressed_outflow_weight),
            ValuationMethod=pl.lit(valuation_method),
            ValuationCurve=pl.lit(valuation_curve),
            ReferenceRate=pl.lit(reference_rate),
            CouponFrequency=pl.lit(coupon_frequency),
            AccountingMethod=pl.lit(accounting_method),
            Book=pl.lit("old"),
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
            AccruedInterestWeight=interest_accrual(
                pl.lit(1.0),
                pl.col("InterestRate"),
                pl.col("PreviousCouponDate"),
                pl.col("NextCouponDate"),
                current_date,
            )
        )
    )

    # Handle quantity calculation differently for notional vs book_value instruments
    if notionals is not None:
        # For notional instruments (e.g., swaps), scale notionals to match target book value
        # Book value formula: sum(Quantity * CleanPrice + Quantity * AccruedInterestWeight)
        # Therefore: sum(Quantity * (CleanPrice + AccruedInterestWeight)) = book_value
        # Scale Notional proportionally: Quantity = Notional * scale_factor
        # where scale_factor ensures: sum(Notional * scale_factor * (CleanPrice + AccruedInterestWeight)) = book_value
        denominator = (pl.col("Notional") * (pl.col("CleanPrice") + pl.col("AccruedInterestWeight"))).sum()
        df = df.with_columns(
            Quantity=pl.when(denominator != 0.0).then(book_value / denominator * pl.col("Notional")).otherwise(0.0)
        ).drop("Notional")
    else:
        # For book_value instruments, derive Quantity from BookValue
        # For amortized cost: BookValue = Quantity + Agio + AccruedInterest + Impairment
        # For fair value: BookValue = Quantity * CleanPrice + AccruedInterest + Agio
        # Both can be expressed as: BookValue = Quantity * (1 + AgioWeight + AccruedInterestWeight - CoverageRate)
        df = df.with_columns(
            Quantity=pl.col("BookValue")
            / (1 + pl.col("AgioWeight") + pl.col("AccruedInterestWeight") - pl.col("CoverageRate"))
        ).drop("BookValue")

    df = df.with_columns(
        Impairment=-pl.col("Quantity") * pl.col("CoverageRate"),
        AccruedInterest=pl.col("Quantity") * pl.col("AccruedInterestWeight"),
        Agio=pl.col("Quantity") * pl.col("AgioWeight"),
        Undrawn=pl.col("Quantity") * pl.col("UndrawnPortion"),
        OtherOffBalanceWeight=other_off_balance_weight,
        ReferenceRate=pl.col("ReferenceRate").cast(pl.String),
        ValuationCurve=pl.col("ValuationCurve").cast(pl.String),
        CleanPrice=pl.col("CleanPrice").cast(pl.Float64),
    ).with_columns(
        FloatingRate=curves.floating_rate_expr(), Spread=pl.col("InterestRate") - curves.floating_rate_expr()
    )

    # Perform valuation to initialize the t0 valuation error
    zero_rates = market_rates.curves.get_zero_rates()
    valuation_method_object = ValuationMethodRegistry.get(valuation_method)
    df = valuation_method_object.calculated_dirty_price(df, current_date, zero_rates, "CalculatedPrice")
    df = df.with_columns(
        valuation_method_object.valuation_error(
            pl.col("CalculatedPrice"), pl.col("CleanPrice") + pl.col("AccruedInterestWeight")
        ).alias("ValuationError")
    ).drop(["AgioWeight", "AccruedInterestWeight", "UndrawnPortion", "CoverageRate", "CalculatedPrice"])

    positions = Positions(df)

    positions.validate()

    # Always validate book_value matches target (works for both notional and non-notional instruments)
    generated_bv = positions.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
    assert abs(generated_bv - book_value) < 1e-2, f"Generated book value not equal to target {book_value}"

    return positions


def generate_random_numbers(number: int, minimum: float, maximum: float, mean: float) -> list:
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


def create_synthetic_balance_sheet(
    current_date: datetime.date,
    scenario: Scenario,
    config_path: str | None = os.path.join(EXAMPLE_FOLDER, "example_bs.csv"),
    config_table: pl.DataFrame = None,
) -> BalanceSheet:
    # Iterate over synthetic_data.csv using polars to create each of the items

    if config_table is None:
        config_table = pl.read_csv(config_path)

    curves = generate_synthetic_curves()

    market_rates = scenario.market_data.get_market_rates(current_date)

    positions = []
    for row in config_table.iter_rows(named=True):
        position_input = {name: read_range(value) if name.endswith("_range") else value for name, value in row.items()}
        if row["number"] > 0:
            positions.append(
                generate_synthetic_positions(
                    market_rates=market_rates, current_date=current_date, curves=curves, **position_input
                )
            )

    combined_positions = Positions.combine(*positions)

    bs = BalanceSheet(combined_positions._data, current_date)

    return bs


def read_range(value: str) -> tuple | None:
    if value is None or len(value.strip()) == 0:
        return None
    else:
        return tuple(float(x.strip()) for x in value.strip("()").split(","))
