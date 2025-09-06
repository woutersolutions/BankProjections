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


def generate_synthetic_assets(
    total_amount: Optional[float] = None,
    num_positions: int = 100,
    asset_types: Optional[list[str]] = None,
    currencies: Optional[list[str]] = None,
    valuation_methods: Optional[list[str]] = None,
    random_seed: Optional[int] = None,
    **additional_columns: Any,
) -> Positions:
    """
    Generate synthetic asset positions with business rules for assets.

    Assets can have:
    - Agio (premium/discount)
    - Impairment provisions
    - Both amortized cost and fair value valuation
    - Maturity dates, coupon rates, etc.

    Args:
        total_amount: Total quantity across all positions. If None, uses random amounts.
        num_positions: Number of positions to generate
        asset_types: List of asset types to randomly assign
        currencies: List of currencies to randomly assign
        valuation_methods: List of valuation methods to randomly assign
        random_seed: Seed for reproducible random generation
        **additional_columns: Any additional columns to include with their values

    Returns:
        Positions object with synthetic asset data
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Default asset types
    if asset_types is None:
        asset_types = ["loan", "bond", "cash", "securities"]
    if currencies is None:
        currencies = ["EUR", "USD", "GBP"]
    if valuation_methods is None:
        valuation_methods = ["amortized cost", "fair value"]

    # Generate quantities (positive for assets)
    if total_amount is not None:
        weights = [random.expovariate(1) for _ in range(num_positions)]
        total_weight = sum(weights)
        quantities = [total_amount * w / total_weight for w in weights]
    else:
        quantities = [random.uniform(1000, 1000000) for _ in range(num_positions)]

    # Generate dates for assets
    base_date = datetime.date(2025, 1, 1)
    next_coupon_dates = [
        base_date.replace(month=random.randint(1, 12), day=random.randint(1, 28)) for _ in range(num_positions)
    ]
    coupon_freqs = [random.choice(["Monthly", "Quarterly", "Annual"]) for _ in range(num_positions)]
    interest_rates = [round(random.uniform(0.01, 0.10), 4) for _ in range(num_positions)]
    # Maturity: 1-10 years from base date
    maturity_dates = [base_date.replace(year=base_date.year + random.randint(1, 10)) for _ in range(num_positions)]

    data = {
        "Quantity": quantities,
        "Impairment": [0.0] * num_positions,  # Start with zero for balanced sheet
        "AccruedInterest": [0.0] * num_positions,  # Start with zero for balanced sheet
        "CleanPrice": [1.0] * num_positions,  # At par for balanced sheet
        "Agio": [0.0] * num_positions,  # Start with zero for balanced sheet
        "AssetType": [random.choice(asset_types) for _ in range(num_positions)],
        "Currency": [random.choice(currencies) for _ in range(num_positions)],
        "ValuationMethod": [random.choice(valuation_methods) for _ in range(num_positions)],
        "InterestRate": interest_rates,
        "NextCouponDate": next_coupon_dates,
        "CouponFrequency": coupon_freqs,
        "MaturityDate": maturity_dates,
        "PrepaymentRate": [round(random.uniform(0.0, 0.10), 4) for _ in range(num_positions)],
        "IsAccumulating": [random.choice([True, False]) for _ in range(num_positions)],
        "RedemptionType": [random.choice(["bullet", "annuity", "linear"]) for _ in range(num_positions)],
        "BalanceSheetSide": ["Asset"] * num_positions,
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


def generate_synthetic_liabilities(
    total_amount: Optional[float] = None,
    num_positions: int = 100,
    liability_types: Optional[list[str]] = None,
    currencies: Optional[list[str]] = None,
    random_seed: Optional[int] = None,
    **additional_columns: Any,
) -> Positions:
    """
    Generate synthetic liability positions with business rules for liabilities.

    Liabilities have different rules:
    - No agio (premium/discount)
    - No impairment provisions
    - Typically use amortized cost valuation
    - Have maturity dates and interest rates
    - Negative quantities (funding side)

    Args:
        total_amount: Total quantity across all positions (will be made negative)
        num_positions: Number of positions to generate
        liability_types: List of liability types to randomly assign
        currencies: List of currencies to randomly assign
        random_seed: Seed for reproducible random generation
        **additional_columns: Any additional columns to include with their values

    Returns:
        Positions object with synthetic liability data
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Default liability types
    if liability_types is None:
        liability_types = ["deposit", "borrowing", "debt_security"]
    if currencies is None:
        currencies = ["EUR", "USD", "GBP"]

    # Generate quantities (negative for liabilities)
    if total_amount is not None:
        weights = [random.expovariate(1) for _ in range(num_positions)]
        total_weight = sum(weights)
        quantities = [-total_amount * w / total_weight for w in weights]
    else:
        quantities = [-random.uniform(1000, 1000000) for _ in range(num_positions)]

    # Generate dates for liabilities
    base_date = datetime.date(2025, 1, 1)
    next_coupon_dates = [
        base_date.replace(month=random.randint(1, 12), day=random.randint(1, 28)) for _ in range(num_positions)
    ]
    coupon_freqs = [random.choice(["Monthly", "Quarterly", "Annual"]) for _ in range(num_positions)]
    interest_rates = [round(random.uniform(0.01, 0.08), 4) for _ in range(num_positions)]  # Typically lower than assets
    # Maturity: 1-10 years from base date
    maturity_dates = [base_date.replace(year=base_date.year + random.randint(1, 10)) for _ in range(num_positions)]

    data = {
        "Quantity": quantities,
        "Impairment": [0.0] * num_positions,  # No impairment for liabilities
        "AccruedInterest": [0.0] * num_positions,  # Start with zero for balanced sheet
        "CleanPrice": [1.0] * num_positions,  # Typically at par
        "Agio": [0.0] * num_positions,  # No agio for liabilities
        "AssetType": [random.choice(liability_types) for _ in range(num_positions)],
        "Currency": [random.choice(currencies) for _ in range(num_positions)],
        "ValuationMethod": ["amortized cost"] * num_positions,  # Typically amortized cost
        "InterestRate": interest_rates,
        "NextCouponDate": next_coupon_dates,
        "CouponFrequency": coupon_freqs,
        "MaturityDate": maturity_dates,
        "PrepaymentRate": [round(random.uniform(0.0, 0.05), 4) for _ in range(num_positions)],  # Lower prepayment
        "IsAccumulating": [False] * num_positions,  # Liabilities typically don't accumulate
        "RedemptionType": [
            random.choice(["bullet", "linear"]) for _ in range(num_positions)
        ],  # Simpler redemption types for liabilities
        "BalanceSheetSide": ["Liability"] * num_positions,
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


def generate_synthetic_equity(
    total_amount: Optional[float] = None,
    num_positions: int = 100,
    equity_types: Optional[list[str]] = None,
    currencies: Optional[list[str]] = None,
    random_seed: Optional[int] = None,
    **additional_columns: Any,
) -> Positions:
    """
    Generate synthetic equity positions with business rules for equity.

    Equity has specific rules:
    - No maturity date (perpetual)
    - No coupon payments or frequency
    - No agio or impairment
    - Always amortized cost valuation
    - Negative quantities (funding side)

    Args:
        total_amount: Total quantity across all positions (will be made negative)
        num_positions: Number of positions to generate
        equity_types: List of equity types to randomly assign
        currencies: List of currencies to randomly assign
        random_seed: Seed for reproducible random generation
        **additional_columns: Any additional columns to include with their values

    Returns:
        Positions object with synthetic equity data
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Default equity types
    if equity_types is None:
        equity_types = ["common_stock", "retained_earnings", "tier1_capital"]
    if currencies is None:
        currencies = ["EUR", "USD", "GBP"]

    # Generate quantities (negative for equity)
    if total_amount is not None:
        weights = [random.expovariate(1) for _ in range(num_positions)]
        total_weight = sum(weights)
        quantities = [-total_amount * w / total_weight for w in weights]
    else:
        quantities = [-random.uniform(1000, 1000000) for _ in range(num_positions)]

    data = {
        "Quantity": quantities,
        "Impairment": [0.0] * num_positions,  # No impairment for equity
        "AccruedInterest": [0.0] * num_positions,  # No accrued interest for equity
        "CleanPrice": [1.0] * num_positions,  # At par
        "Agio": [0.0] * num_positions,  # No agio for equity
        "AssetType": [random.choice(equity_types) for _ in range(num_positions)],
        "Currency": [random.choice(currencies) for _ in range(num_positions)],
        "ValuationMethod": ["amortized cost"] * num_positions,  # Always amortized cost
        "InterestRate": [0.0] * num_positions,  # No interest rate for equity
        "NextCouponDate": [None] * num_positions,  # No coupon dates for equity
        "CouponFrequency": ["Never"] * num_positions,  # No coupon frequency for equity
        "MaturityDate": [None] * num_positions,  # No maturity for equity
        "PrepaymentRate": [0.0] * num_positions,  # No prepayment for equity
        "IsAccumulating": [False] * num_positions,  # Equity doesn't accumulate
        "RedemptionType": ["perpetual"] * num_positions,  # Equity is perpetual
        "BalanceSheetSide": ["Equity"] * num_positions,
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
    assets = generate_synthetic_assets(
        total_amount=total_assets,
        num_positions=num_asset_positions,
        asset_types=["loan", "bond", "cash", "securities"],
        random_seed=random_seed,
        **generation_kwargs,
    )

    # Generate liabilities
    total_liabilities = total_assets * asset_liability_ratio
    liabilities = generate_synthetic_liabilities(
        total_amount=total_liabilities,
        num_positions=num_liability_positions,
        liability_types=["deposit", "borrowing", "debt_security"],
        random_seed=random_seed + 1 if random_seed else None,
        **generation_kwargs,
    )

    # Generate equity
    total_equity = total_assets * equity_ratio
    equity = generate_synthetic_equity(
        total_amount=total_equity,
        num_positions=num_equity_positions,
        equity_types=["common_stock", "retained_earnings"],
        random_seed=random_seed + 2 if random_seed else None,
        **generation_kwargs,
    )

    combined_positions = Positions.combine(assets, liabilities, equity)
    cash_account = BalanceSheetItem(AssetType="cash")
    pnl_account = BalanceSheetItem(BalanceSheetSide="Equity")

    return BalanceSheet(combined_positions._data, cash_account, pnl_account)


def generate_synthetic_positions(
    total_amount: Optional[float] = None,
    num_positions: int = 100,
    asset_types: Optional[list[str]] = None,
    currencies: Optional[list[str]] = None,
    valuation_methods: Optional[list[str]] = None,
    balance_sheet_side: str = "asset",
    random_seed: Optional[int] = None,
    **additional_columns: Any,
) -> Positions:
    """
    Legacy function for backward compatibility.
    Delegates to appropriate specialized functions based on balance_sheet_side.
    """
    if balance_sheet_side.lower() == "asset":
        return generate_synthetic_assets(
            total_amount=total_amount,
            num_positions=num_positions,
            asset_types=asset_types,
            currencies=currencies,
            valuation_methods=valuation_methods,
            random_seed=random_seed,
            **additional_columns,
        )
    elif balance_sheet_side.lower() == "liability":
        return generate_synthetic_liabilities(
            total_amount=total_amount,
            num_positions=num_positions,
            liability_types=asset_types,  # Use asset_types parameter as liability_types
            currencies=currencies,
            random_seed=random_seed,
            **additional_columns,
        )
    elif balance_sheet_side.lower() == "equity":
        return generate_synthetic_equity(
            total_amount=total_amount,
            num_positions=num_positions,
            equity_types=asset_types,  # Use asset_types parameter as equity_types
            currencies=currencies,
            random_seed=random_seed,
            **additional_columns,
        )
    else:
        raise ValueError(f"Invalid balance_sheet_side: {balance_sheet_side}. Must be 'asset', 'liability', or 'equity'")
