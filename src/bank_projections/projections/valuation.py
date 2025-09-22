from bank_projections.projections.base_registry import BaseRegistry


class ValuationMethod:
    pass


class ValuationRegistry(BaseRegistry[ValuationMethod]):
    pass


ValuationRegistry.register("Amortized cost", ValuationMethod())
ValuationRegistry.register("Fair Value", ValuationMethod())
