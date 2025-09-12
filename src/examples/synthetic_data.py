import datetime
import os
import random
from typing import Optional

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, BalanceSheetItem, Positions
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.frequency import FrequencyRegistry
from examples import EXAMPLE_FOLDER


def generate_synthetic_positions(
    book_value: float,
    number: int,
    balance_sheet_side: str,
    item_type: str,
    valuation_method: str,
    redemption_type: str,
    coupon_frequency: str,
    current_date: datetime.date,
    currency: str = "EUR",
    coverage_rate_range: Optional[tuple[float, float]] = None,
    interest_rate_range: Optional[tuple[float, float]] = None,
    agio_range: Optional[tuple[float, float]] = None,
    prepayment_rate: Optional[float] = 0.0,
    minimum_maturity: Optional[int] = None,
    maximum_maturity: Optional[int] = None,
    accumulating: Optional[bool] = False,
) -> Positions:
    # Generate random book values that sum to the target book value
    if book_value == 0 or number == 1:
        book_values = [book_value] * number
    elif book_value > 0:
        book_values = generate_random_numbers(number, 1, book_value * min(0.9, (100.0 / number)), book_value / number)
    else:
        book_values = generate_random_numbers(number, book_value / number, -book_value * min(0.9, (100.0 / number)), -1)
    # Scale so that the total book value matches exactly
    book_values = [value * book_value / sum(book_values) for value in book_values]

    if agio_range is None:
        agios = [0.0] * number
    else:
        agios = generate_random_numbers(number, agio_range[0], agio_range[1], (agio_range[0] + agio_range[1]) / 2)

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

    if redemption_type == "perpetual":
        maturity_dates = [None] * number
    elif redemption_type in ["bullet", "linear", "annuity"]:
        # Generate uniform random based on current date and min/max maturity in years, by adding a random number of days
        maturity_dates = [
            current_date + datetime.timedelta(days=random.randint(minimum_maturity * 365, maximum_maturity * 365))
            for _ in range(number)
        ]
    else:
        raise ValueError(f"Unknown redemption type: {redemption_type}")

    match coupon_frequency.lower():
        case "daily":
            maximum_next_coupon_days = 1
        case "weekly":
            maximum_next_coupon_days = 7
        case "monthly":
            maximum_next_coupon_days = 30  # Average days in a month
        case "quarterly":
            maximum_next_coupon_days = 91  # Average days in a quarter
        case "semiAnnual":
            maximum_next_coupon_days = 182  # Average days in half a year
        case "annual":
            maximum_next_coupon_days = 365  # Average days in a year
        case "never":
            maximum_next_coupon_days = None
        case _:
            raise ValueError(f"Unknown coupon frequency: {coupon_frequency}")
        # Generate next coupon dates within the next coupon period

    if maximum_next_coupon_days is None:
        next_coupon_dates = [None] * number
    else:
        next_coupon_dates = [
            current_date + datetime.timedelta(days=random.randint(1, maximum_next_coupon_days)) for _ in range(number)
        ]

    # Create polars dataframe with all the calculated fields
    df = pl.DataFrame(
        {
            "BookValue": book_values,
            "CoverageRate": coverage_rates,
            "CleanPrice": [1.0] * number,  # At par for balanced sheet
            "AgioWeight": agios,
            "ItemType": [item_type] * number,
            "Currency": [currency] * number,
            "ValuationMethod": [valuation_method] * number,
            "InterestRate": interest_rates,
            "NextCouponDate": next_coupon_dates,
            "CouponFrequency": [coupon_frequency] * number,
            "MaturityDate": maturity_dates,
            "PrepaymentRate": [prepayment_rate] * number,
            "IsAccumulating": [accumulating] * number,
            "RedemptionType": [redemption_type] * number,
            "BalanceSheetSide": [balance_sheet_side] * number,
        },
        schema_overrides={"NextCouponDate": pl.Date},
    )

    df = (
        df.with_columns(
            AccruedInterestWeight=FrequencyRegistry.portion_year()
            * FrequencyRegistry.portion_passed(pl.col("NextCouponDate"), current_date)
            * pl.col("InterestRate")
        )
        .with_columns(
            Quantity=pl.col("BookValue")
            / (1 + pl.col("AgioWeight") + pl.col("AccruedInterestWeight") - pl.col("CoverageRate"))
        )
        .with_columns(
            Impairment=-pl.col("Quantity") * pl.col("CoverageRate"),
            AccruedInterest=pl.col("Quantity") * pl.col("AccruedInterestWeight"),
            Agio=pl.col("Quantity") * pl.col("AgioWeight"),
        )
        .drop(["AgioWeight", "AccruedInterestWeight", "CoverageRate", "BookValue"])
    )

    positions = Positions(df)

    positions.validate()
    assert abs(positions.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value) - book_value) < 1e-2, (
        f"Generated book value not equal to target {book_value}"
    )

    return positions


def generate_random_numbers(number: int, minimum: float, maximum: float, mean: float) -> list:
    # Use beta distribution to generate numbers
    if mean <= minimum:
        raise ValueError(f"Mean {mean} must be greater than minimum {minimum}")
    if mean >= maximum:
        raise ValueError(f"Mean {mean} must be less than maximum {maximum}")

    alpha = ((mean - minimum) / (maximum - minimum)) * 5
    beta = ((maximum - mean) / (maximum - minimum)) * 5
    random_numbers = [random.betavariate(alpha, beta) * (maximum - minimum) + minimum for _ in range(number)]

    return random_numbers


def create_synthetic_balance_sheet(
    current_date: datetime.date,
    config_path: Optional[str] = os.path.join(EXAMPLE_FOLDER, "knab_bs.csv"),
    config_table: pl.DataFrame = None,
) -> BalanceSheet:
    # Iterate over synthetic_data.csv using polars to create each of the items

    if config_table is None:
        config_table = pl.read_csv(config_path)

    positions = []
    for row in config_table.iter_rows(named=True):
        position_input = {name: read_range(value) if name.endswith("_range") else value for name, value in row.items()}
        if row["number"] > 0:
            positions.append(generate_synthetic_positions(current_date=current_date, **position_input))

    combined_positions = Positions.combine(*positions)

    cash_account = BalanceSheetItem(ItemType="Cash")
    pnl_account = BalanceSheetItem(ItemType="Unaudited earnings")

    bs = BalanceSheet(combined_positions._data, cash_account, pnl_account)

    return bs


def read_range(value: str) -> Optional[tuple]:
    if value is None or len(value.strip()) == 0:
        return None
    else:
        return tuple(float(x.strip()) for x in value.strip("()").split(","))
