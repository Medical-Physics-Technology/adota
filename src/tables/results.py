import logging
from typing import Dict, List, Optional

import numpy as np


def print_results_table(
    energies: List[float],
    gprs_calc: List[float],
    rmses: List[float],
    mapes_0_1_pct: List[float],
    mapes_1_pct: List[float],
    mapes_5_pct: List[float],
    mapes_10_pct: List[float],
    rdes: List[float],
    gamma_params: Dict[str, float],
    beamlet_angles: List[List[float]] = None,
    logger: Optional[logging.Logger] = None,
):
    """Display an aligned evaluation table in the terminal and log.

    Parameters
    ----------
    energies : list[float]
        Energy values [MeV].
    gprs_calc : list[float]
        Gamma passing rates corresponding to each energy.
    rmses : list[float]
        Root mean square errors [Gy].
    mapes_0_1_pct, mapes_1_pct, mapes_5_pct, mapes_10_pct : list[float]
        Thresholded MAPE values [%] using GT-dose masks at 0.1%, 1%, 5%, 10%.
    gamma_params : dict
        Dictionary with gamma evaluation parameters.
    logger : logging.Logger, optional
        If provided, each line is also logged via logger.info().
    """

    def _print(line: str) -> None:
        print(line, flush=True)
        if logger is not None:
            logger.info(line)

    header = (
        "Energies [MeV] | GPR ({}%, {}mm, {}%) | RMSE [Gy] | "
        "MAPE@0.1% GT [%] | MAPE@1% GT [%] | MAPE@5% GT [%] | "
        "MAPE@10% GT [%] | RDE [%]"
    ).format(
        gamma_params["dose_percent_threshold"],
        gamma_params["distance_mm_threshold"],
        gamma_params["lower_percent_dose_cutoff"],
    )
    _print(header)
    _print("=" * len(header))

    if beamlet_angles is not None:
        sorted_data = sorted(
            zip(
                energies,
                beamlet_angles,
                gprs_calc,
                rmses,
                mapes_0_1_pct,
                mapes_1_pct,
                mapes_5_pct,
                mapes_10_pct,
                rdes,
            ),
            key=lambda x: x[0],
        )
        col_widths = [12, 17, 10, 14, 16, 15, 16, 17, 10]
        header_fmt = (
            f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | "
            f"{{:<{col_widths[2]}}} | {{:<{col_widths[3]}}} | "
            f"{{:<{col_widths[4]}}} | {{:<{col_widths[5]}}} | "
            f"{{:<{col_widths[6]}}} | {{:<{col_widths[7]}}} | "
            f"{{:<{col_widths[8]}}} |"
        )
        row_fmt = (
            f"| {{:>{col_widths[0]}.2f}} | {{:>{col_widths[1]}}} | "
            f"{{:>{col_widths[2]}.2f}} | {{:>{col_widths[3]}.9f}} | "
            f"{{:>{col_widths[4]}.4f}} | {{:>{col_widths[5]}.4f}} | "
            f"{{:>{col_widths[6]}.4f}} | {{:>{col_widths[7]}.4f}} | "
            f"{{:>{col_widths[8]}.4f}} |"
        )
        _print(
            header_fmt.format(
                "Energy [MeV]",
                "beamlet angles",
                "GPR [%]",
                "RMSE [Gy]",
                "MAPE@0.1% [%]",
                "MAPE@1% [%]",
                "MAPE@5% [%]",
                "MAPE@10% [%]",
                "RDE [%]",
            )
        )
        separator_width = sum(col_widths) + 3 * len(col_widths) + 1
        _print("-" * separator_width)
        for e, angles, gpr, rmse, mape_0_1, mape_1, mape_5, mape_10, rde in sorted_data:
            angle_str = f"{angles[0]:.2f}, {angles[1]:.2f}"
            _print(
                row_fmt.format(
                    e,
                    angle_str,
                    gpr,
                    rmse,
                    mape_0_1,
                    mape_1,
                    mape_5,
                    mape_10,
                    rde,
                )
            )
    else:
        sorted_data = sorted(
            zip(
                energies,
                gprs_calc,
                rmses,
                mapes_0_1_pct,
                mapes_1_pct,
                mapes_5_pct,
                mapes_10_pct,
                rdes,
            ),
            key=lambda x: x[0],
        )
        col_widths = [12, 10, 14, 16, 15, 16, 17, 10]
        header_fmt = (
            f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | "
            f"{{:<{col_widths[2]}}} | {{:<{col_widths[3]}}} | "
            f"{{:<{col_widths[4]}}} | {{:<{col_widths[5]}}} | "
            f"{{:<{col_widths[6]}}} | {{:<{col_widths[7]}}} |"
        )
        row_fmt = (
            f"| {{:>{col_widths[0]}.2f}} | {{:>{col_widths[1]}.2f}} | "
            f"{{:>{col_widths[2]}.9f}} | {{:>{col_widths[3]}.4f}} | "
            f"{{:>{col_widths[4]}.4f}} | {{:>{col_widths[5]}.4f}} | "
            f"{{:>{col_widths[6]}.4f}} | {{:>{col_widths[7]}.4f}} |"
        )
        _print(
            header_fmt.format(
                "Energy [MeV]",
                "GPR [%]",
                "RMSE [Gy]",
                "MAPE@0.1% [%]",
                "MAPE@1% [%]",
                "MAPE@5% [%]",
                "MAPE@10% [%]",
                "RDE [%]",
            )
        )
        separator_width = sum(col_widths) + 3 * len(col_widths) + 1
        _print("-" * separator_width)
        for e, gpr, rmse, mape_0_1, mape_1, mape_5, mape_10, rde in sorted_data:
            _print(row_fmt.format(e, gpr, rmse, mape_0_1, mape_1, mape_5, mape_10, rde))

    _print("=" * separator_width)

    summary_metrics = [
        ("GPR", gprs_calc, "{:.2f} %"),
        ("RMSE", rmses, "{:.9f} Gy"),
        ("MAPE@0.1% GT", mapes_0_1_pct, "{:.4f} %"),
        ("MAPE@1% GT", mapes_1_pct, "{:.4f} %"),
        ("MAPE@5% GT", mapes_5_pct, "{:.4f} %"),
        ("MAPE@10% GT", mapes_10_pct, "{:.4f} %"),
        ("RDE", rdes, "{:.4f} %"),
    ]

    for label, values, value_fmt in summary_metrics:
        mean = value_fmt.format(np.mean(values))
        std = value_fmt.format(np.std(values))
        min_val = value_fmt.format(np.min(values))
        max_val = value_fmt.format(np.max(values))
        _print(f"Mean {label}: {mean} ± {std}")
        _print(f"Min {label}: {min_val}")
        _print(f"Max {label}: {max_val}")


def render_comparison_table(
    row_labels: List[str],
    column_headers: List[str],
    cells: List[List[str]],
    *,
    row_header: str = "Run",
) -> str:
    """Render an aligned ``Row x Column`` text table (also valid Markdown).

    Generic helper for the cross-run validation experiment: one row per run,
    one column per metric, each cell a pre-formatted string (e.g. ``mean ± std``).
    The output is column-aligned monospace text whose ``|``-delimited rows and
    separator line also parse as a GitHub Markdown table.

    Args:
        row_labels: Left-column label for each row (e.g. run names).
        column_headers: Metric column headers.
        cells: ``cells[i][j]`` is the string for row ``i``, column ``j``.
            Must be a rectangular ``len(row_labels) x len(column_headers)`` grid.
        row_header: Header for the left-most (label) column.

    Returns:
        The table as a single string (rows joined by newlines).
    """
    n_rows, n_cols = len(row_labels), len(column_headers)
    if len(cells) != n_rows or any(len(r) != n_cols for r in cells):
        raise ValueError(
            f"cells must be {n_rows}x{n_cols}; got "
            f"{len(cells)}x{[len(r) for r in cells]}"
        )

    headers = [row_header, *column_headers]
    grid = [headers] + [
        [row_labels[i], *cells[i]] for i in range(n_rows)
    ]
    widths = [
        max(len(grid[r][c]) for r in range(len(grid))) for c in range(n_cols + 1)
    ]

    def _fmt_row(values: List[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[c]) for c, v in enumerate(values)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [_fmt_row(headers), sep]
    lines += [_fmt_row(grid[r]) for r in range(1, len(grid))]
    return "\n".join(lines)
