# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BankProjections is a Python-based balance sheet forecasting tool for banks, used for capital management, liquidity management, and stress testing. The project uses AGPL v3 license.

## Architecture

### Backend Components

The backend consists of five main components located in `src/bank_projections/`:

1. **data_preparation**: Constructs bank financial statements from input data (SQL, YAML, CSV, or Excel)
2. **amortization**: Handles amortization calculations for balance sheet items
3. **projections**: Performs scenario-based projections with user-defined time intervals
4. **metrics**: Computes financial metrics (capital ratios, ROC, LCR, NSFR, etc.)
5. **output**: Stores and formats results

### Project Structure

```
src/bank_projections/
├── data_preparation/   # Input parsing and financial statement construction
├── amortization/       # Amortization logic for balance sheet items  
├── projections/        # Scenario application and time-series projections
├── metrics/           # Financial metrics calculations
└── output/            # Result storage and formatting

tests/                 # Unit and integration tests
docs/                  # Documentation
examples/              # Example scenarios and synthetic data generation
```

## Development Commands

### Testing
```bash
pytest                          # Run all tests
pytest tests/unit/             # Run unit tests only
pytest -v                      # Verbose output
pytest --cov=src              # Run with coverage
pytest -k "test_name"         # Run specific test
```

### Code Quality
```bash
ruff check .                   # Check linting issues
ruff format .                  # Format code (120 char line width)
mypy src/                     # Type checking
```

### Development Workflow
```bash
python -m pip install -e .     # Install package in development mode
python -m bank_projections     # Run the application
```

## Development Guidelines

### Code Style
- Use ruff for formatting with 120 character line width (configured in pyproject.toml)
- Use mypy for type checking with strict mode
- Only use doc strings and comments in code when an explanation is needed
- All functions and classes must have proper type hints
- Use docstrings for all modules, classes, and public functions

### Data Models
- Use Pydantic for input validation and data models
- Define clear interfaces between components
- Scenario inputs should be validated against schema

### Testing Requirements
- All calculations must have comprehensive unit tests
- Test edge cases: negative rates, zero balances, extreme time periods
- Use pytest fixtures for test data
- Generate synthetic data for testing

### Import Conventions
- Use `from bank_projections.module import Class` for imports within the codebase
- The `src/` directory is in the Python path, so `bank_projections` can be imported directly
- Avoid `src.bank_projections` import style

## Important Reminders

1. **No Real Data**: This repo uses only synthetic data for testing and demonstration
2. **Scenarios**: Support multiple concurrent scenarios in projections
3. **Input Formats**: Support YAML, CSV, and Excel for scenario definitions
4. **Metrics**: Implement standard regulatory metrics (Basel III ratios, liquidity metrics)

## Synthetic Data Generation

The `examples/` directory should contain utilities to generate realistic but synthetic bank data including:
- Sample balance sheets
- Scenario definitions
- Stress test scenarios
- Example configuration files