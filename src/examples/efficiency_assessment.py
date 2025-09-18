"""Efficiency assessment script for Bank Projections.

This script measures performance across different scenarios:
1. Varying time horizons (number of time steps)
2. Varying balance sheet sizes (number of positions)

It generates performance graphs showing processing time vs. parameters.
"""

import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from loguru import logger

from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.time import TimeHorizon
from examples import EXAMPLE_FOLDER, OUTPUT_FOLDER
from examples.synthetic_data import create_synthetic_balance_sheet


class EfficiencyAssessment:
    """Class to assess efficiency of bank projections across different parameters."""

    def __init__(
        self,
        output_dir: str = OUTPUT_FOLDER,
        synthetic_data_config: str = os.path.join(EXAMPLE_FOLDER, "knab_bs.csv"),
        size_multipliers=(1, 5, 10, 100),
        number_of_projections=(1, 5, 10, 20, 50, 100),
    ):
        """Initialize efficiency assessment.

        Args:
            output_dir: Directory to save results and graphs
        """
        self.output_dir = Path(output_dir)

        # Configuration paths
        self.synthetic_data_config = synthetic_data_config

        # Size multipliers to test
        self.size_multipliers = size_multipliers
        self.number_of_projections = number_of_projections

    def measure_time_horizon_performance(self) -> list[dict]:
        """Measure performance across different time horizons."""
        logger.info("Starting time horizon performance assessment")

        # Results storage
        time_horizon_results: list[dict] = []
        self.output_dir.mkdir(exist_ok=True)

        # Base configuration for balance sheet
        start_date = datetime.date(2024, 12, 31)
        base_bs = create_synthetic_balance_sheet(current_date=start_date, config_path=str(self.synthetic_data_config))

        rules = [Runoff()]

        for n in self.number_of_projections:
            # Create time horizon
            horizon = TimeHorizon.from_numbers(
                start_date=start_date,
                number_of_months=n - 1,
                end_of_month=True,
            )

            num_time_steps = len(horizon)
            logger.info(f"Testing time horizon: {num_time_steps} steps")

            # Measure performance
            projection = Projection(rules, horizon)

            start_time = time.perf_counter()
            _ = projection.run(base_bs)
            end_time = time.perf_counter()

            processing_time = end_time - start_time

            # Store results
            time_horizon_results.append(
                {
                    "description": str(num_time_steps) + " steps",
                    "num_time_steps": num_time_steps,
                    "processing_time": processing_time,
                }
            )

            logger.info(
                f"Completed in {processing_time:.3f}s with {num_time_steps} time steps and {len(base_bs)} positions"
            )

        return time_horizon_results

    def measure_balance_sheet_size_performance(self) -> list[dict]:
        """Measure performance across different balance sheet sizes."""
        logger.info("Starting balance sheet size performance assessment")

        balance_sheet_results = []

        start_date = datetime.date(2024, 12, 31)

        # Fixed time horizon for all balance sheet size tests
        horizon = TimeHorizon.from_numbers(
            start_date=start_date,
            number_of_days=7,
            number_of_months=12 * 4,
            end_of_month=True,
        )

        rules = [Runoff()]

        for multiplier in self.size_multipliers:
            logger.info(f"Testing balance sheet size multiplier: {multiplier}")

            # Create modified balance sheet with increased size
            bs = self._create_scaled_balance_sheet(start_date, multiplier)

            # Calculate actual number of positions
            num_positions = self._count_balance_sheet_positions(bs)
            logger.info(f"Balance sheet positions: {num_positions}")

            # Measure performance
            projection = Projection(rules, horizon)

            start_time = time.perf_counter()
            _ = projection.run(bs)
            end_time = time.perf_counter()

            processing_time = end_time - start_time

            # Store results
            balance_sheet_results.append(
                {
                    "size_multiplier": multiplier,
                    "num_positions": num_positions,
                    "processing_time": processing_time,
                }
            )

            logger.info(
                f"Completed in {processing_time:.3f}s with {len(horizon)} time steps balance sheets and {num_positions} positions"
            )

        return balance_sheet_results

    def _create_scaled_balance_sheet(self, current_date: datetime.date, multiplier: int):
        """Create a balance sheet scaled by the given multiplier.

        Args:
            current_date: Date to use for balance sheet creation
            multiplier: Factor to multiply the 'number' column by

        Returns:
            BalanceSheet with scaled size
        """
        # Read and modify the config CSV
        config_df = pl.read_csv(str(self.synthetic_data_config))

        # Scale the 'number' column
        scaled_config_df = config_df.with_columns((pl.col("number") * multiplier).alias("number"))

        # Create balance sheet with scaled config
        bs = create_synthetic_balance_sheet(current_date=current_date, config_table=scaled_config_df)

        return bs

    def _count_balance_sheet_positions(self, balance_sheet) -> int:
        """Count the total number of positions in a balance sheet.

        Args:
            balance_sheet: BalanceSheet object

        Returns:
            Total number of positions
        """
        # Access the internal data DataFrame
        return len(balance_sheet._data)

    def create_visualizations(self, time_horizon_results, balance_sheet_results) -> None:
        """Create and save performance visualization graphs."""
        logger.info("Creating performance visualizations")

        try:
            # Set up the plotting style
            plt.style.use("default")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Graph 1: Time Horizon Performance
            if time_horizon_results:
                time_steps = [r["num_time_steps"] for r in time_horizon_results]
                processing_times = [r["processing_time"] for r in time_horizon_results]
                descriptions = [r["description"] for r in time_horizon_results]

                ax1.scatter(time_steps, processing_times, alpha=0.7, s=60)
                ax1.plot(time_steps, processing_times, "-", alpha=0.5)

                ax1.set_xlabel("Number of Time Steps")
                ax1.set_ylabel("Processing Time (seconds)")
                ax1.set_title("Performance vs. Time Horizon")
                ax1.grid(True, alpha=0.3)

                # Add annotations for some points
                for i, desc in enumerate(descriptions):
                    if i % 3 == 0:  # Annotate every 3rd point to avoid clutter
                        ax1.annotate(
                            desc,
                            (time_steps[i], processing_times[i]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                            alpha=0.7,
                        )
            else:
                ax1.text(0.5, 0.5, "No time horizon data", transform=ax1.transAxes, ha="center")

            # Graph 2: Balance Sheet Size Performance
            if balance_sheet_results:
                num_positions = [r["num_positions"] for r in balance_sheet_results]
                processing_times = [r["processing_time"] for r in balance_sheet_results]

                ax2.scatter(num_positions, processing_times, alpha=0.7, s=60, color="orange")
                ax2.plot(num_positions, processing_times, "-", alpha=0.5, color="orange")

                ax2.set_xlabel("Number of Balance Sheet Positions")
                ax2.set_ylabel("Processing Time (seconds)")
                ax2.set_title("Performance vs. Balance Sheet Size")
                ax2.grid(True, alpha=0.3)

                # Add multiplier labels
                multipliers = [r["size_multiplier"] for r in balance_sheet_results]
                for i, mult in enumerate(multipliers):
                    ax2.annotate(
                        f"{mult}x",
                        (num_positions[i], processing_times[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )
            else:
                ax2.text(0.5, 0.5, "No balance sheet size data", transform=ax2.transAxes, ha="center")

            plt.tight_layout()

            # Save the graph
            graph_path = self.output_dir / "efficiency_assessment.png"
            plt.savefig(graph_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved performance graphs to {graph_path}")

            # Close the figure to free memory
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            raise

    def save_results_to_csv(self, results, file_name) -> None:
        """Save detailed results to CSV files."""
        logger.info(f"Saving detailed results to CSV {file_name}")

        time_horizon_df = pd.DataFrame(results)
        time_horizon_path = self.output_dir / "time_horizon_performance.csv"
        time_horizon_df.to_csv(time_horizon_path, index=False)
        logger.info(f"Saved time horizon results to {time_horizon_path}")

    def print_summary(self, time_horizon_results, balance_sheet_results) -> None:
        """Print a summary of the performance assessment."""
        logger.info("=== EFFICIENCY ASSESSMENT SUMMARY ===")

        if time_horizon_results:
            logger.info(f"\nTime Horizon Performance ({len(time_horizon_results)} tests):")
            min_time = min(r["processing_time"] for r in time_horizon_results)
            max_time = max(r["processing_time"] for r in time_horizon_results)
            min_steps = min(r["num_time_steps"] for r in time_horizon_results)
            max_steps = max(r["num_time_steps"] for r in time_horizon_results)

            logger.info(f"  Time steps range: {min_steps} - {max_steps}")
            logger.info(f"  Processing time range: {min_time:.3f}s - {max_time:.3f}s")
            logger.info(f"  Performance ratio: {max_time / min_time:.2f}x slower for largest horizon")

        if balance_sheet_results:
            logger.info(f"\nBalance Sheet Size Performance ({len(balance_sheet_results)} tests):")
            min_time = min(r["processing_time"] for r in balance_sheet_results)
            max_time = max(r["processing_time"] for r in balance_sheet_results)
            min_positions = min(r["num_positions"] for r in balance_sheet_results)
            max_positions = max(r["num_positions"] for r in balance_sheet_results)

            logger.info(f"  Position count range: {min_positions} - {max_positions}")
            logger.info(f"  Processing time range: {min_time:.3f}s - {max_time:.3f}s")
            logger.info(f"  Performance ratio: {max_time / min_time:.2f}x slower for largest balance sheet")

    def run_full_assessment(self) -> None:
        """Run the complete efficiency assessment."""
        logger.info("Starting complete efficiency assessment")

        # Run both performance tests
        time_horizon_results = self.measure_time_horizon_performance()
        balance_sheet_results = self.measure_balance_sheet_size_performance()

        # Generate outputs
        self.create_visualizations(time_horizon_results, balance_sheet_results)
        self.save_results_to_csv(balance_sheet_results, "balance_sheet_performance.csv")
        self.save_results_to_csv(time_horizon_results, "time_horizon_performance.csv")
        self.print_summary(time_horizon_results, balance_sheet_results)

        logger.info("Efficiency assessment completed!")


def main():
    # Run the assessment
    assessment = EfficiencyAssessment()
    assessment.run_full_assessment()


if __name__ == "__main__":
    main()
