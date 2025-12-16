import os

import yaml

from bank_projections.output_config import OutputConfig
from bank_projections.projections.accrual import Accrual
from bank_projections.projections.agio_redemption import AgioRedemption
from bank_projections.projections.coupon_payment import CouponPayment
from bank_projections.projections.projection import Projection
from bank_projections.projections.redemption import Redemption
from bank_projections.projections.valuation import Valuation
from bank_projections.scenarios.audit import AuditRule
from bank_projections.scenarios.cost_income import CostIncomeRule
from bank_projections.scenarios.excel_sheet_format import TemplateTypeRegistry
from bank_projections.scenarios.mutation import BalanceSheetMutationRule
from bank_projections.scenarios.production import ProductionRule
from bank_projections.scenarios.scenario import Scenario, ScenarioConfig
from bank_projections.scenarios.tax import TaxProjectionRule
from bank_projections.utils.logging import setup_logger_format_with_context
from bank_projections.utils.time import TimeHorizon
from examples import EXAMPLE_FOLDER
from examples.synthetic_data import create_synthetic_balance_sheet

if __name__ == "__main__":
    setup_logger_format_with_context()

    with open(EXAMPLE_FOLDER / "scenario_config.yaml") as f:
        scenario_config = ScenarioConfig(**yaml.safe_load(f))

    excel_inputs = TemplateTypeRegistry.load_paths(scenario_config.input_paths)
    scenario = Scenario(excel_inputs)
    rules = {
        "Coupons": CouponPayment(),
        "Redemption": Redemption(),
        "Accrual": Accrual(),
        "Agio": AgioRedemption(),
        "Valuation": Valuation(),
        "Mutations": BalanceSheetMutationRule(),
        "Cost/Income": CostIncomeRule(),
        "Production": ProductionRule(),
        "Tax": TaxProjectionRule(),
        "Audit": AuditRule(),
    }

    horizon = TimeHorizon.from_config(scenario_config.time_horizon)

    start_bs = create_synthetic_balance_sheet(horizon.start_date, scenario)

    with open(EXAMPLE_FOLDER / "output_config.yaml") as f:
        output_config = OutputConfig(**yaml.safe_load(f))

    projection = Projection({"base": scenario}, horizon, rules)

    result = projection.run(start_bs, output_config.aggregation)
    result.to_excel(os.path.join(output_config.output_folder, output_config.output_file), open_after=True)
