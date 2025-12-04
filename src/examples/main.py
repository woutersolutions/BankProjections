import os

import yaml

from bank_projections.config import ScenarioConfig
from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.valuation import Valuation
from bank_projections.scenarios.template_registry import TemplateRegistry
from bank_projections.utils.logging import setup_logger_format_with_context
from bank_projections.utils.time import TimeHorizon
from examples import EXAMPLE_FOLDER, OUTPUT_FOLDER
from examples.synthetic_data import create_synthetic_balance_sheet

if __name__ == "__main__":
    setup_logger_format_with_context()

    with open(EXAMPLE_FOLDER / "scenario_config.yaml") as f:
        scenario_config = ScenarioConfig(**yaml.safe_load(f))

    scenario = TemplateRegistry.load_paths(scenario_config.rule_paths)
    scenario.rules = {"Runoff": Runoff(), "Valuation": Valuation(), **scenario.rules}

    horizon = TimeHorizon.from_config(scenario_config.time_horizon)

    start_bs = create_synthetic_balance_sheet(horizon.start_date, scenario)
    projection = Projection({"base": scenario}, horizon)
    result = projection.run(start_bs)
    result.to_excel(os.path.join(OUTPUT_FOLDER, "main example.xlsx"), open_after=True)
