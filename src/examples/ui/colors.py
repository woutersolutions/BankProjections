PRIMARY_COLORS = {
    "deep_navy": "#1a365d",
    "medium_blue": "#2c5282",
    "bright_blue": "#3182ce",
}

SEMANTIC_COLORS = {
    "success_green": "#48bb78",
    "warning_orange": "#ed8936",
    "error_red": "#e74c3c",
    "info_purple": "#5a67d8",
}

NEUTRAL_COLORS = {
    "dark_gray": "#2d3748",
    "medium_gray": "#4a5568",
    "light_gray": "#718096",
    "background_light": "#f7fafc",
    "background_white": "#ffffff",
    "border_gray": "#cbd5e0",
    "surface_gray": "#e2e8f0",
}


def get_chart_colors() -> list[str]:
    return [
        SEMANTIC_COLORS["info_purple"],
        PRIMARY_COLORS["bright_blue"],
        SEMANTIC_COLORS["success_green"],
        SEMANTIC_COLORS["warning_orange"],
        PRIMARY_COLORS["medium_blue"],
        SEMANTIC_COLORS["error_red"],
        PRIMARY_COLORS["deep_navy"],
    ]
