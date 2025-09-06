import datetime

import polars as pl

from bank_projections.projections.frequency import Annual, FrequencyRegistry
from bank_projections.projections.redemption import (
    AnnuityRedemption,
    BulletRedemption,
    LinearRedemption,
    PerpetualRedemption,
    RedemptionRegistry,
)


class TestBulletRedemption:
    def test_bullet_before_maturity(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 12, 31))
        projection = datetime.date(2025, 6, 30)

        factor = BulletRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 0.0

    def test_bullet_at_maturity(self):
        maturity = pl.lit(datetime.date(2025, 6, 30))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 6, 30))
        projection = datetime.date(2025, 6, 30)

        factor = BulletRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 1.0

    def test_bullet_after_maturity(self):
        maturity = pl.lit(datetime.date(2025, 6, 30))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 6, 30))
        projection = datetime.date(2025, 12, 31)

        factor = BulletRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 1.0


class TestPerpetualRedemption:
    def test_perpetual_no_redemption(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 12, 31))
        projection = datetime.date(2025, 6, 30)

        factor = PerpetualRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 0.0


class TestLinearRedemption:
    def test_linear_constant_principal_payment(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2023, 12, 31))
        projection = datetime.date(2023, 12, 31)  # Start of loan

        # Create test dataframe with required columns
        df = pl.DataFrame({"CouponFrequency": ["annual"], "dummy": [1]}).with_columns(
            [
                LinearRedemption.redemption_factor(maturity, rate, coupon_date, projection).alias("factor"),
                FrequencyRegistry.number_due(coupon_date, pl.lit(projection)).alias("payments_left"),
            ]
        )

        payments_left = df["payments_left"][0]
        expected_factor = 1.0 / payments_left if payments_left > 0 else 0.0

        assert abs(df["factor"][0] - expected_factor) < 1e-10


class TestAnnuityRedemption:
    def test_annuity_payment_formula(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2023, 12, 31))
        projection = datetime.date(2023, 12, 31)  # Start of loan

        # Create test dataframe with required columns
        df = pl.DataFrame({"CouponFrequency": ["annual"], "dummy": [1]}).with_columns(
            [
                AnnuityRedemption.redemption_factor(maturity, rate, coupon_date, projection).alias("factor"),
                FrequencyRegistry.number_due(coupon_date, pl.lit(projection)).alias("payments_left"),
                FrequencyRegistry.portion_year().alias("period_fraction"),
            ]
        )

        payments_left = df["payments_left"][0]
        period_rate = 0.05 * df["period_fraction"][0]
        expected_factor = period_rate / (1 - (1 + period_rate) ** (-payments_left)) if payments_left > 0 else 0.0

        assert abs(df["factor"][0] - expected_factor) < 1e-10


class TestRedemptionRegistry:
    def test_registry_routing(self):
        # Set up frequency registry for testing
        FrequencyRegistry.register("annual", Annual)

        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 12, 31))
        projection = datetime.date(2025, 6, 30)

        # Test bullet routing
        df_bullet = pl.DataFrame({"RedemptionType": ["bullet"], "CouponFrequency": ["annual"]}).with_columns(
            RedemptionRegistry.redemption_factor(maturity, rate, coupon_date, projection).alias("factor")
        )

        # Test perpetual routing
        df_perpetual = pl.DataFrame({"RedemptionType": ["perpetual"], "CouponFrequency": ["annual"]}).with_columns(
            RedemptionRegistry.redemption_factor(maturity, rate, coupon_date, projection).alias("factor")
        )

        assert df_bullet["factor"][0] == 0.0  # Before maturity
        assert df_perpetual["factor"][0] == 0.0  # Never redeems


class TestRedemptionValidation:
    def test_bullet_redemption_validation(self):
        # Test that bullet redemption validates successfully (no additional columns needed)
        df = pl.DataFrame({"RedemptionType": ["bullet"]}).with_columns(
            BulletRedemption.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_annuity_redemption_validation_valid(self):
        # Test valid CouponFrequency for annuity redemption
        df = pl.DataFrame({"RedemptionType": ["annuity"], "CouponFrequency": ["Annual"]}).with_columns(
            AnnuityRedemption.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_annuity_redemption_validation_invalid(self):
        # Test invalid CouponFrequency for annuity redemption
        df = pl.DataFrame({"RedemptionType": ["annuity"], "CouponFrequency": ["Invalid"]}).with_columns(
            AnnuityRedemption.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is False

    def test_annuity_redemption_validation_null(self):
        # Test null CouponFrequency for annuity redemption
        df = pl.DataFrame({"RedemptionType": ["annuity"], "CouponFrequency": [None]}).with_columns(
            AnnuityRedemption.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is False

    def test_linear_redemption_validation_valid(self):
        # Test valid CouponFrequency for linear redemption
        df = pl.DataFrame({"RedemptionType": ["linear"], "CouponFrequency": ["Quarterly"]}).with_columns(
            LinearRedemption.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_perpetual_redemption_validation(self):
        # Test that perpetual redemption validates successfully (no additional columns needed)
        df = pl.DataFrame({"RedemptionType": ["perpetual"]}).with_columns(
            PerpetualRedemption.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_registry_validation_valid_bullet(self):
        # Set up frequency registry for testing
        FrequencyRegistry.register("annual", Annual)

        # Test registry validation for bullet type
        df = pl.DataFrame({"RedemptionType": ["bullet"]}).with_columns(
            RedemptionRegistry.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_registry_validation_valid_annuity(self):
        # Test registry validation for annuity type with valid frequency
        df = pl.DataFrame({"RedemptionType": ["annuity"], "CouponFrequency": ["Annual"]}).with_columns(
            RedemptionRegistry.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_registry_validation_null_redemption_type(self):
        # Test registry validation with null RedemptionType
        df = pl.DataFrame({"RedemptionType": [None], "CouponFrequency": ["Annual"]}).with_columns(
            RedemptionRegistry.required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is False

    def test_validate_specific_bullet_redemption(self):
        # Test specific bullet redemption validation
        df = pl.DataFrame({"dummy": [1]}).with_columns(
            RedemptionRegistry.get("bullet").required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_validate_specific_annuity_redemption_valid(self):
        # Test specific annuity redemption validation with valid frequency
        df = pl.DataFrame({"CouponFrequency": ["Annual"]}).with_columns(
            RedemptionRegistry.get("annuity").required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is True

    def test_validate_specific_annuity_redemption_invalid(self):
        # Test specific annuity redemption validation with invalid frequency
        df = pl.DataFrame({"CouponFrequency": ["Invalid"]}).with_columns(
            RedemptionRegistry.get("annuity").required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is False

    def test_validate_unknown_redemption_type(self):
        # Test validation of unknown redemption type
        df = pl.DataFrame({"dummy": [1]}).with_columns(
            RedemptionRegistry.get("unknown").required_columns_validation().alias("is_valid")
        )
        assert df["is_valid"][0] is False
