from bank_projections.projections.base_registry import BaseRegistry


class AccountingMethod:
    pass


class AccountingMethodRegistry(BaseRegistry[AccountingMethod]):
    pass


AccountingMethodRegistry.register("Amortized cost", AccountingMethod())
AccountingMethodRegistry.register("Fair Value", AccountingMethod())
