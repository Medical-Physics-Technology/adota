import logging
from typing import List, Dict, Optional
import numpy as np


def print_results_table(
    energies: List[float],
    gprs_calc: List[float],
    rmses: List[float],
    mapes: List[float],
    rdes: List[float],
    gamma_params: Dict[str, float],
    beamlet_angles: List[List[float]] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Display a neatly aligned results table for energies and related metrics in the terminal.

    Parameters
    ----------
    energies : list[float]
        Energy values [MeV].
    gprs_calc : list[float]
        Gamma passing rates corresponding to each energy.
    rmses : list[float]
        Root mean square errors [eV/g / proton].
    mapes : list[float]
        Mean absolute percentage errors [%].
    gamma_params : dict
        Dictionary with gamma evaluation parameters:
            - 'dose_percent_threshold' : float
            - 'distance_mm_threshold' : float
            - 'lower_percent_dose_cutoff' : float
    logger : logging.Logger, optional
        If provided, each line is also logged via logger.info().
    """

    def _print(line: str) -> None:
        """Print to console and optionally log."""
        print(line, flush=True)
        if logger is not None:
            logger.info(line)

    header = "Energies [MeV] | GPR ({}%, {}mm, {}%) | RMSE [eV/g / proton] | MAPE [%] | RDE [%]".format(
        gamma_params["dose_percent_threshold"],
        gamma_params["distance_mm_threshold"],
        gamma_params["lower_percent_dose_cutoff"],
    )
    _print(header)
    _print("=" * len(header))

    # Sort all metrics by energy
    if beamlet_angles is not None:
        sorted_data = sorted(
            zip(energies, beamlet_angles, gprs_calc, rmses, mapes, rdes),
            key=lambda x: x[0],
        )
    else:
        sorted_data = sorted(
            zip(energies, gprs_calc, rmses, mapes, rdes), key=lambda x: x[0]
        )

    # Define consistent column widths
    if beamlet_angles is not None:
        col_widths = [14, 17, 20, 24, 14, 14]
        header_fmt = f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | {{:<{col_widths[2]}}} | {{:<{col_widths[3]}}} | {{:<{col_widths[4]}}} | {{:<{col_widths[5]}}} |"
        row_fmt = f"| {{:>{col_widths[0]}.2f}} | {{:>{col_widths[1]}}} | {{:>{col_widths[2]}.2f}} | {{:>{col_widths[3]}.9f}} | {{:>{col_widths[4]}.4f}} | {{:>{col_widths[5]}.4f}} |"

        # Print header row
        _print(
            header_fmt.format(
                "Energy [MeV]",
                "beamlet angles",
                "GPR [%]",
                "RMSE [Gy]",
                "MAPE [%]",
                "RDE [%]",
            )
        )
        _print("-" * (sum(col_widths) + 16))  # account for pipes and spaces

        # Print data rows
        for e, angles, gpr, rmse, mape, rde in sorted_data:
            angle_str = f"{angles[0]:.2f}, {angles[1]:.2f}"
            _print(row_fmt.format(e, angle_str, gpr, rmse, mape, rde))
    else:
        col_widths = [14, 20, 24, 14, 14]
        header_fmt = f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | {{:<{col_widths[2]}}} | {{:<{col_widths[3]}}} | {{:<{col_widths[4]}}} |"
        row_fmt = f"| {{:>{col_widths[0]}.2f}} | {{:>{col_widths[1]}.2f}} | {{:>{col_widths[2]}.9f}} | {{:>{col_widths[3]}.4f}} | {{:>{col_widths[4]}.4f}} |"

        # Print header row
        _print(
            header_fmt.format(
                "Energy [MeV]", "GPR [%]", "RMSE [Gy]", "MAPE [%]", "RDE [%]"
            )
        )
        _print("-" * (sum(col_widths) + 13))  # account for pipes and spaces

        # Print data rows
        for e, gpr, rmse, mape, rde in sorted_data:
            _print(row_fmt.format(e, gpr, rmse, mape, rde))

    # Calculate separator width based on whether angles are included
    separator_width = (
        (sum(col_widths) + 16) if beamlet_angles is not None else (sum(col_widths) + 13)
    )
    _print("=" * separator_width)
    # Print mean values, std values, min and max:
    mean_gpr = np.mean(gprs_calc)
    std_gpr = np.std(gprs_calc)
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    mean_mape = np.mean(mapes)
    std_mape = np.std(mapes)
    mean_rde = np.mean(rdes)
    std_rde = np.std(rdes)
    _print(f"Mean GPR: {mean_gpr:.2f} % ± {std_gpr:.2f} %")
    _print(f"Mean RMSE: {mean_rmse:.9f} Gy ± {std_rmse:.9f} Gy")
    _print(f"Mean MAPE: {mean_mape:.4f} % ± {std_mape:.4f} %")
    _print(f"Mean RDE: {mean_rde:.4f} % ± {std_rde:.4f} %")
    _print(f"Min GPR: {np.min(gprs_calc):.2f} %")
    _print(f"Max GPR: {np.max(gprs_calc):.2f} %")
    _print(f"Min RMSE: {np.min(rmses):.9f} Gy")
    _print(f"Max RMSE: {np.max(rmses):.9f} Gy")
    _print(f"Min MAPE: {np.min(mapes):.4f} %")
    _print(f"Max MAPE: {np.max(mapes):.4f} %")
    _print(f"Min RDE: {np.min(rdes):.4f} %")
    _print(f"Max RDE: {np.max(rdes):.4f} %")
