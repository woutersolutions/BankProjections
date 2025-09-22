import datetime
from typing import Any

import numpy as np

from bank_projections.config import Config
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.template import AmountRuleBase
from bank_projections.utils.parsing import clean_identifier, get_identifier, is_in_identifiers, read_bool, read_date


class BalanceSheetMutationRule(AmountRuleBase):
    def __init__(self, rule_input: dict[str, Any], amount: float):
        self.amount = amount
        self.relative = True
        self.multiplicative = False
        self.offset_liquidity = False
        self.offset_pnl = False
        self.reason = MutationReason(rule="BalanceSheetMutationRule")
        self.date: datetime.date | None = None

        if is_in_identifiers("item", list(rule_input.keys())):
            value = rule_input[get_identifier("item", list(rule_input.keys()))]
            if value in ["", np.nan, None]:
                self.item = BalanceSheetItem()
            else:
                self.item = BalanceSheetItemRegistry.get(value)
        else:
            self.item = BalanceSheetItem()
        if is_in_identifiers("counter item", list(rule_input.keys())):
            value = rule_input[get_identifier("counter item", list(rule_input.keys()))]
            if value in ["", np.nan, None]:
                self.counter_item = None
            else:
                self.counter_item = BalanceSheetItemRegistry.get(value)
        else:
            self.counter_item = None

        for key, value in rule_input.items():
            match clean_identifier(key):
                case _ if value in ["", np.nan, None]:
                    pass
                case "item" | "counteritem":
                    pass
                case "metric":
                    self.metric = BalanceSheetMetrics.get(value)
                case _ if key.startswith("counter"):
                    label = clean_identifier(key[len("counter") :])
                    if is_in_identifiers(label, Config.BALANCE_SHEET_LABELS):
                        if self.counter_item is None:
                            self.counter_item = BalanceSheetItem(**{label: value})
                        else:
                            self.counter_item = self.counter_item.add_identifier(label, value)
                    else:
                        raise KeyError(f"{key} not recognized as valid balance sheet label")
                case _ if is_in_identifiers(key, Config.BALANCE_SHEET_LABELS):
                    self.item = self.item.add_identifier(key, value)
                case "relative":
                    self.relative = read_bool(value)
                case "multiplicative":
                    self.multiplicative = read_bool(value)
                case "offsetliquidity":
                    self.offset_liquidity = read_bool(value)
                case "offsetpnl":
                    self.offset_pnl = read_bool(value)
                case "date":
                    self.date = read_date(value)
                case _:
                    raise KeyError(f"{key} not recognized in BalanceSheetMutationRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        if self.date is None or increment.contains(self.date):
            bs.mutate_metric(
                self.item,
                self.metric,
                self.amount,
                self.reason,
                self.relative,
                self.multiplicative,
                self.offset_liquidity,
                self.offset_pnl,
                self.counter_item,
            )

        bs.validate()

        return bs
