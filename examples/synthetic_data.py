import datetime
import random
from dataclasses import dataclass
from typing import Any, Optional

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, BalanceSheetItem, Positions


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    total_assets: Optional[float] = None
    num_positions: int = 100
    asset_types: list[str] = None
    currencies: list[str] = None
    valuation_methods: list[str] = None
    coverage_rate_range: tuple[float, float] = (0.0, 0.05)
    agio_weight_range: tuple[float, float] = (-0.02, 0.02)
    accrued_interest_range: tuple[float, float] = (0.0, 0.01)
    dirty_price_range: tuple[float, float] = (-0.05, 0.05)
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.asset_types is None:
            self.asset_types = ["loan", "bond", "deposit", "equity"]
        if self.currencies is None:
            self.currencies = ["EUR", "USD", "GBP"]
        if self.valuation_methods is None:
            self.valuation_methods = ["amortized cost", "fair value"]


def generate_synthetic_positions(
    total_amount: Optional[float] = None,
    num_positions: int = 100,
    asset_types: Optional[list[str]] = None,
    currencies: Optional[list[str]] = None,
    valuation_methods: Optional[list[str]] = None,
    coverage_rate_range: tuple[float, float] = (0.0, 0.05),
    agio_weight_range: tuple[float, float] = (-0.02, 0.02),
    accrued_interest_range: tuple[float, float] = (0.0, 0.01),
    dirty_price_range: tuple[float, float] = (-0.05, 0.05),
    balance_sheet_side: str = "asset",
    random_seed: Optional[int] = None,
    **additional_columns: Any,
) -> Positions:
    """
    Generate synthetic position data for testing and demonstration.

    Args:
        total_amount: Total quantity across all positions. If None, uses random amounts.
        num_positions: Number of positions to generate
        asset_types: List of asset types to randomly assign
        currencies: List of currencies to randomly assign
        valuation_methods: List of valuation methods to randomly assign
        coverage_rate_range: Min/max coverage rate values
        agio_weight_range: Min/max agio weight values
        accrued_interest_range: Min/max accrued interest rate values
        dirty_price_range: Min/max dirty price values
        balance_sheet_side: "asset" or "liability" to determine sign
        random_seed: Seed for reproducible random generation
        **additional_columns: Any additional columns to include with their values

    Returns:
        Positions object with synthetic data
    """
    if random_seed is not None:
        random.seed(random_seed)

    config = SyntheticDataConfig(
        total_assets=total_amount,
        num_positions=num_positions,
        asset_types=asset_types,
        currencies=currencies,
        valuation_methods=valuation_methods,
        coverage_rate_range=coverage_rate_range,
        agio_weight_range=agio_weight_range,
        accrued_interest_range=accrued_interest_range,
        dirty_price_range=dirty_price_range,
        random_seed=random_seed,
    )

    # Generate quantities
    if total_amount is not None:
        # Distribute total amount across positions using exponential distribution
        weights = [random.expovariate(1) for _ in range(num_positions)]
        total_weight = sum(weights)
        quantities = [total_amount * w / total_weight for w in weights]
    else:
        quantities = [random.uniform(1000, 1000000) for _ in range(num_positions)]

    # Apply sign based on balance sheet side
    sign = 1 if balance_sheet_side == "asset" else -1
    quantities = [q * sign for q in quantities]

    # Generate other attributes with simpler values for balance sheet balancing
    # Ensure InterestRate, CouponDate, CouponFrequency, MaturityDate are always present
    start_year = 2020
    start_month = 1
    coupon_dates = [datetime.date(start_year, start_month, 1) for _ in range(num_positions)]
    coupon_freqs = [random.choice(["M", "Q", "Y"]) for _ in range(num_positions)]
    interest_rates = [round(random.uniform(0.01, 0.10), 4) for _ in range(num_positions)]
    # Maturity: 1-10 years after coupon date
    maturity_dates = [
        coupon_dates[i].replace(year=coupon_dates[i].year + random.randint(1, 10)) for i in range(num_positions)
    ]

    data = {
        "Quantity": quantities,
        "Impairment": [0.0] * num_positions,  # Required for coverage_rate metric
        "AccruedInterest": [0.0] * num_positions,  # Required for accrued_interest metric
        "CleanPrice": [1.0] * num_positions,  # Required for fair value calculation
        "Agio": [0.0] * num_positions,  # Required for book_value calculation (was AgioWeight but this is the amount)
        "AssetType": [random.choice(config.asset_types) for _ in range(num_positions)],
        "Currency": [random.choice(config.currencies) for _ in range(num_positions)],
        "ValuationMethod": [random.choice(config.valuation_methods) for _ in range(num_positions)],
        "InterestRate": interest_rates,
        "CouponDate": coupon_dates,
        "CouponFrequency": coupon_freqs,
        "MaturityDate": maturity_dates,
    }

    # Add any additional columns
    for col_name, col_value in additional_columns.items():
        if isinstance(col_value, list):
            if len(col_value) != num_positions:
                raise ValueError(f"Additional column '{col_name}' must have {num_positions} values")
            data[col_name] = col_value
        else:
            data[col_name] = [col_value] * num_positions

    df = pl.DataFrame(data)
    return Positions(df)


def create_balanced_balance_sheet(
    total_assets: float = 10_000_000,
    asset_liability_ratio: float = 0.9,
    equity_ratio: float = 0.1,
    num_asset_positions: int = 50,
    num_liability_positions: int = 30,
    num_equity_positions: int = 5,
    random_seed: Optional[int] = None,
    **generation_kwargs: Any,
) -> BalanceSheet:
    """
    Create a balanced synthetic balance sheet with assets, liabilities, and equity.

    Args:
        total_assets: Total asset amount
        asset_liability_ratio: Ratio of liabilities to assets (default 0.9)
        equity_ratio: Ratio of equity to assets (default 0.1)
        num_asset_positions: Number of asset positions
        num_liability_positions: Number of liability positions
        num_equity_positions: Number of equity positions
        random_seed: Seed for reproducible generation
        **generation_kwargs: Additional arguments passed to generate_synthetic_positions

    Returns:
        BalanceSheet that balances (assets = liabilities + equity)
    """
    if abs(asset_liability_ratio + equity_ratio - 1.0) > 0.001:
        raise ValueError("asset_liability_ratio + equity_ratio must equal 1.0")

    # Generate assets
    assets = generate_synthetic_positions(
        total_amount=total_assets,
        num_positions=num_asset_positions,
        balance_sheet_side="asset",
        asset_types=["loan", "bond", "cash", "securities"],
        random_seed=random_seed,
        BalanceSheetSide="Asset",
        **generation_kwargs,
    )

    # Generate liabilities
    total_liabilities = total_assets * asset_liability_ratio
    liabilities = generate_synthetic_positions(
        total_amount=total_liabilities,
        num_positions=num_liability_positions,
        balance_sheet_side="liability",
        asset_types=["deposit", "borrowing", "debt_security"],
        random_seed=random_seed + 1 if random_seed else None,
        BalanceSheetSide="Liability",
        **generation_kwargs,
    )

    # Generate equity
    total_equity = total_assets * equity_ratio
    equity = generate_synthetic_positions(
        total_amount=total_equity,
        num_positions=num_equity_positions,
        balance_sheet_side="liability",
        asset_types=["common_stock", "retained_earnings"],
        random_seed=random_seed + 2 if random_seed else None,
        BalanceSheetSide="Equity",
        **generation_kwargs,
    )

    combined_positions = Positions.combine(assets, liabilities, equity)
    cash_account = BalanceSheetItem(AssetType="cash")
    pnl_account = BalanceSheetItem(BalanceSheetSide="Equity")

    return BalanceSheet(combined_positions._data, cash_account, pnl_account)
