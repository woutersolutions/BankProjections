import datetime
from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.utils.base_registry import BaseRegistry


class RedemptionType(ABC):
    @classmethod
    @abstractmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        pass

    @classmethod
    @abstractmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Returns an expression that validates all required columns are available and have valid values.
        :param date:
        """
        pass


class RedemptionTypeRegistry(BaseRegistry[RedemptionType], RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        expr = pl.lit(0.0)
        for name, redemption_cls in cls.stripped_items.items():
            expr = (
                pl.when(pl.col("RedemptionType") == name)
                .then(redemption_cls.redemption_factor(maturity_date, interest_rate, coupon_date, projection_date))
                .otherwise(expr)
            )
        return expr

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Returns an expression that validates all required columns for all registered redemption types.
        :param date:
        """
        # Base requirement: RedemptionType column must exist and be non-null
        result = [pl.col("RedemptionType").is_in(RedemptionTypeRegistry.stripped_names())]
        for name, redemption_cls in cls.stripped_items.items():
            for expr in redemption_cls.required_columns_validation(date):
                result.append(pl.when(pl.col("RedemptionType") == name).then(expr).otherwise(True))

        return result

    @classmethod
    def validate_df(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        date: datetime.date,
        sample_rows: int = 10,
        id_cols: list[str] | None = None,
    ) -> None:
        """Validates that the DataFrame has all required columns for the registered redemption types."""
        rules = cls.required_columns_validation(date)
        validate_df(df, rules, sample_rows=sample_rows, id_cols=id_cols)


class BulletRedemptionType(RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.when(maturity_date <= pl.lit(projection_date)).then(pl.lit(1.0)).otherwise(pl.lit(0.0))

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Validates that bullet redemption only needs maturity date.
        :param date:
        """
        return [pl.col("MaturityDate").is_not_null()]


class AnnuityRedemptionType(RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        payments_done = FrequencyRegistry.number_due(coupon_date, pl.lit(projection_date))
        total_payments = FrequencyRegistry.number_due(coupon_date, pl.col("MaturityDate"))
        period_rate = interest_rate * FrequencyRegistry.portion_year()

        return (
            pl.when(maturity_date <= pl.lit(projection_date))
            .then(pl.lit(1.0))
            .when(payments_done <= 0)
            .then(pl.lit(0.0))
            .when(interest_rate == 0)
            .then(payments_done / total_payments)
            .otherwise(((1 + period_rate).pow(payments_done) - 1) / ((1 + period_rate).pow(total_payments) - 1))
        )

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Validates that CouponFrequency column exists and has valid values for annuity redemption.
        :param date:
        """
        # Check if CouponFrequency column exists and has valid values
        return [
            pl.col("MaturityDate").is_not_null(),
            pl.col("CouponFrequency").is_not_null(),
            pl.col("CouponFrequency").is_in(list(FrequencyRegistry.stripped_names())),
            pl.col("NextCouponDate").is_not_null() | (pl.col("MaturityDate") <= pl.lit(date)),
            pl.col("InterestRate").is_not_null(),
        ]


class PerpetualRedemptionType(RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.lit(0.0)

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Validates that perpetual redemption needs no additional columns.
        :param date:
        """
        return [pl.col("MaturityDate").is_null()]


class LinearRedemptionType(RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        payments_left = FrequencyRegistry.number_due(coupon_date, pl.lit(projection_date))

        return (
            pl.when(maturity_date <= pl.lit(projection_date))
            .then(pl.lit(1.0))
            .when(payments_left <= 0)
            .then(pl.lit(0.0))
            .otherwise(pl.lit(1.0) / payments_left)
        )

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Validates that CouponFrequency column exists and has valid values for linear redemption.
        :param date:
        """
        return [pl.col("MaturityDate").is_not_null()]


class NotionalOnlyRedemptionType(RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.when(maturity_date <= pl.lit(projection_date)).then(pl.lit(1.0)).otherwise(pl.lit(0.0))

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Validates that notional redemption only needs maturity date.
        :param date:
        """
        return [pl.col("MaturityDate").is_not_null()]


class ManualRedemptionType(RedemptionType):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.lit(None, pl.Float64)  # Is overridden by manual input

    @classmethod
    def required_columns_validation(cls, date: datetime.date) -> list[pl.Expr]:
        """Validates that manual redemption only needs maturity date.
        :param date:
        """
        return []


# Register all redemption types
RedemptionTypeRegistry.register("bullet", BulletRedemptionType())
RedemptionTypeRegistry.register("annuity", AnnuityRedemptionType())
RedemptionTypeRegistry.register("perpetual", PerpetualRedemptionType())
RedemptionTypeRegistry.register("linear", LinearRedemptionType())
RedemptionTypeRegistry.register("notional", NotionalOnlyRedemptionType())
RedemptionTypeRegistry.register("manual", ManualRedemptionType())


def validate_df(
    df: pl.DataFrame | pl.LazyFrame,
    rules: Sequence[pl.Expr],
    sample_rows: int = 10,
    id_cols: list[str] | None = None,
) -> None:
    """
    rules: list of boolean pl.Expr objects (True = passes).
    Raises ValueError if any rule has at least one failing row.
    """

    # Generate names from expression strings
    names = [str(expr) for expr in rules]

    # Count failures in one select
    fail_exprs = [
        (~expr.fill_null(False)).cast(pl.UInt32).sum().alias(name) for expr, name in zip(rules, names, strict=False)
    ]
    counts_df = df.select(fail_exprs) if isinstance(df, pl.DataFrame) else df.select(fail_exprs).collect()
    counts = counts_df.row(0, named=True)

    failing = {k: v for k, v in counts.items() if v > 0}
    if not failing:
        return  # all good

    # Optional: include a few failing rows per rule
    samples_txt = ""
    if sample_rows and isinstance(df, pl.DataFrame):
        show_cols = id_cols or df.columns
        for expr, name in zip(rules, names, strict=False):
            if counts[name] > 0:
                sample = df.filter(~expr.fill_null(False)).select(show_cols).head(sample_rows)
                samples_txt += f"\n\n{name} — first {min(sample_rows, counts[name])} failing rows:\n{sample}"

    msg_lines = [
        "Validation failed:",
        *[f"- {k}: {v} failures" for k, v in sorted(failing.items(), key=lambda x: (-x[1], x[0]))],
    ]
    if samples_txt:
        msg_lines.append(samples_txt)

    raise ValueError("\n".join(msg_lines))
