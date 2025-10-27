import datetime

import pandas as pd

from examples.ui.colors import NEUTRAL_COLORS, PRIMARY_COLORS, SEMANTIC_COLORS, get_chart_colors


def test_color_configuration() -> None:
    assert isinstance(PRIMARY_COLORS, dict)
    assert "deep_navy" in PRIMARY_COLORS
    assert "medium_blue" in PRIMARY_COLORS
    assert "bright_blue" in PRIMARY_COLORS

    assert isinstance(SEMANTIC_COLORS, dict)
    assert "success_green" in SEMANTIC_COLORS
    assert "warning_orange" in SEMANTIC_COLORS
    assert "error_red" in SEMANTIC_COLORS
    assert "info_purple" in SEMANTIC_COLORS

    assert isinstance(NEUTRAL_COLORS, dict)
    assert "dark_gray" in NEUTRAL_COLORS
    assert "background_white" in NEUTRAL_COLORS


def test_get_chart_colors() -> None:
    colors = get_chart_colors()
    assert isinstance(colors, list)
    assert len(colors) == 7
    assert all(isinstance(color, str) for color in colors)
    assert all(color.startswith("#") for color in colors)


def test_example_model_run_imports() -> None:
    import examples.ui.example_model_run

    assert hasattr(examples.ui.example_model_run, "main")
    assert hasattr(examples.ui.example_model_run, "create_time_series_plot")
    assert hasattr(examples.ui.example_model_run, "create_metrics_plot")


def test_single_asset_run_imports() -> None:
    import examples.ui.single_asset_run

    assert hasattr(examples.ui.single_asset_run, "main")


def test_create_time_series_plot() -> None:
    from examples.ui.example_model_run import create_time_series_plot

    test_data = pd.DataFrame(
        {
            "ProjectionDate": [datetime.date(2024, 12, 31), datetime.date(2025, 1, 31)] * 2,
            "BookValue": [100.0, 110.0, 200.0, 220.0],
            "BalanceSheetSide": ["Assets", "Assets", "Liabilities", "Liabilities"],
        }
    )

    fig = create_time_series_plot(test_data, "ProjectionDate", "BookValue", "BalanceSheetSide", "Test Chart")

    assert fig is not None
    assert fig.layout.title.text == "Test Chart"
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Value"


def test_create_metrics_plot() -> None:
    from examples.ui.example_model_run import create_metrics_plot

    test_data = pd.DataFrame(
        {
            "ProjectionDate": [datetime.date(2024, 12, 31), datetime.date(2025, 1, 31)],
            "CET1Ratio": [0.15, 0.16],
            "LeverageRatio": [0.05, 0.06],
        }
    )

    fig = create_metrics_plot(test_data, ["CET1Ratio", "LeverageRatio"])

    assert fig is not None
    assert fig.layout.title.text == "Selected Metrics Over Time"
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Value"


def test_styles_imports() -> None:
    from examples.ui.styles import apply_custom_styles

    assert callable(apply_custom_styles)
