"""Aggregate ablation study results from multiple training runs.

Reads ``manifest.json`` and ``metrics.jsonl`` from each run directory,
identifies the best epoch per run, and prints a fixed-width comparison
table. Also writes ``results_summary.json`` next to this script.

Usage
-----
    # Glob pattern (quotes required to prevent shell expansion):
    uv run python scripts/ablation/aggregate_results.py 'runs/ablation_*/'

    # Explicit list of dirs:
    uv run python scripts/ablation/aggregate_results.py \\
        runs/ablation_A_... runs/ablation_B_... runs/ablation_C_... runs/ablation_D_...

    # Specify a different output file:
    uv run python scripts/ablation/aggregate_results.py 'runs/ablation_*/' \\
        --output results/my_summary.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False)


# ── Data loading ─────────────────────────────────────────────────────────────


def _load_run(run_dir: Path) -> Optional[dict]:
    """Return a result dict for one run directory, or None on failure."""
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.jsonl"

    if not manifest_path.exists():
        typer.echo(f"  [SKIP] {run_dir}: manifest.json not found", err=True)
        return None
    if not metrics_path.exists():
        typer.echo(f"  [SKIP] {run_dir}: metrics.jsonl not found", err=True)
        return None

    with manifest_path.open() as f:
        manifest = json.load(f)

    cfg = manifest.get("config", {})

    rows = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        typer.echo(f"  [SKIP] {run_dir}: metrics.jsonl is empty", err=True)
        return None

    # Best epoch: minimum val loss_combined_mean.
    best_row = min(
        rows,
        key=lambda r: r.get("val", {}).get("loss_combined_mean", float("inf")),
    )
    val = best_row.get("val", {})

    # GPR: find the epoch closest to best_epoch that has a gpr_mean value.
    best_epoch = best_row.get("epoch", -1)
    gpr_mean: Optional[float] = None
    gpr_epoch: Optional[int] = None
    for row in rows:
        v = row.get("val", {})
        if "gpr_mean" in v and v["gpr_mean"] is not None:
            epoch = row.get("epoch", -1)
            if gpr_mean is None or abs(epoch - best_epoch) < abs(gpr_epoch - best_epoch):
                gpr_mean = v["gpr_mean"]
                gpr_epoch = epoch

    return {
        "run_dir": str(run_dir),
        "config_name": cfg.get("config_name", run_dir.name),
        "flux_mode": cfg.get("flux_mode", "unknown"),
        "loss_mode": cfg.get("loss_mode", "unknown"),
        "best_epoch": best_epoch,
        "n_epochs_completed": len(rows),
        "n_train": manifest.get("n_train"),
        "n_val": manifest.get("n_val"),
        "val_loss": val.get("loss_combined_mean"),
        "rmse_gy_mean": val.get("rmse_gy_mean"),
        "mape_pct_mean": val.get("mape_pct_mean"),
        "rde_pct_mean": val.get("rde_pct_mean"),
        "gpr_mean": gpr_mean,
        "gpr_epoch": gpr_epoch,
    }


# ── Table formatting ──────────────────────────────────────────────────────────


def _fmt(value: Optional[float], fmt: str, scale: float = 1.0) -> str:
    if value is None:
        return "  N/A  "
    return format(value * scale, fmt)


def _delta(value: Optional[float], baseline: Optional[float], scale: float = 1.0) -> str:
    if value is None or baseline is None:
        return "    N/A"
    d = (value - baseline) * scale
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _print_table(results: list) -> None:
    # Baseline: analytical + mse_idd, else first entry.
    baseline = next(
        (r for r in results if r["flux_mode"] == "analytical" and r["loss_mode"] == "mse_idd"),
        results[0],
    )

    col_w = {
        "name": max(len(r["config_name"]) for r in results) + 2,
        "flux": 16,
        "loss": 10,
        "ep": 6,
        "val_loss": 12,
        "rmse": 14,
        "mape": 10,
        "rde": 10,
        "gpr": 10,
        "d_rmse": 12,
        "d_mape": 12,
        "d_rde": 12,
        "d_gpr": 10,
    }

    total_w = sum(col_w.values()) + len(col_w) - 1
    sep = "=" * total_w
    thin = "-" * total_w

    def header() -> str:
        return (
            f"{'Config':<{col_w['name']}} "
            f"{'flux_mode':<{col_w['flux']}} "
            f"{'loss_mode':<{col_w['loss']}} "
            f"{'ep':>{col_w['ep']}} "
            f"{'val_loss':>{col_w['val_loss']}} "
            f"{'rmse_gy(x1e5)':>{col_w['rmse']}} "
            f"{'mape_%':>{col_w['mape']}} "
            f"{'rde_%':>{col_w['rde']}} "
            f"{'gpr_%':>{col_w['gpr']}} "
            f"{'d_rmse':>{col_w['d_rmse']}} "
            f"{'d_mape':>{col_w['d_mape']}} "
            f"{'d_rde':>{col_w['d_rde']}} "
            f"{'d_gpr':>{col_w['d_gpr']}}"
        )

    def data_row(r: dict) -> str:
        marker = " [baseline]" if r is baseline else ""
        return (
            f"{r['config_name'] + marker:<{col_w['name']}} "
            f"{r['flux_mode']:<{col_w['flux']}} "
            f"{r['loss_mode']:<{col_w['loss']}} "
            f"{str(r['best_epoch']):>{col_w['ep']}} "
            f"{_fmt(r['val_loss'], '.6f'):>{col_w['val_loss']}} "
            f"{_fmt(r['rmse_gy_mean'], '.4f', scale=1e5):>{col_w['rmse']}} "
            f"{_fmt(r['mape_pct_mean'], '.3f'):>{col_w['mape']}} "
            f"{_fmt(r['rde_pct_mean'], '.4f'):>{col_w['rde']}} "
            f"{_fmt(r['gpr_mean'], '.3f'):>{col_w['gpr']}} "
            f"{_delta(r['rmse_gy_mean'], baseline['rmse_gy_mean'], scale=1e5):>{col_w['d_rmse']}} "
            f"{_delta(r['mape_pct_mean'], baseline['mape_pct_mean']):>{col_w['d_mape']}} "
            f"{_delta(r['rde_pct_mean'], baseline['rde_pct_mean']):>{col_w['d_rde']}} "
            f"{_delta(r['gpr_mean'], baseline['gpr_mean']):>{col_w['d_gpr']}}"
        )

    typer.echo("")
    typer.echo(sep)
    typer.echo("  ABLATION STUDY RESULTS")
    typer.echo(sep)
    typer.echo(header())
    typer.echo(thin)
    for r in results:
        typer.echo(data_row(r))
    typer.echo(sep)
    typer.echo(
        "  Deltas are (variant - baseline). "
        "RMSE and d_rmse are scaled by 1e5. "
        "GPR epoch shown in summary JSON."
    )
    typer.echo("")


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    run_dirs: List[str] = typer.Argument(
        ...,
        help="Run directories or glob patterns (quote globs to prevent shell expansion).",
    ),
    output: Path = typer.Option(
        Path(__file__).parent / "results_summary.json",
        "--output",
        "-o",
        help="Path for the JSON summary output.",
    ),
) -> None:
    """Aggregate ablation study results and print a comparison table."""
    # Expand any glob patterns.
    expanded: List[Path] = []
    for pattern in run_dirs:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(Path(m) for m in sorted(matches))
        else:
            expanded.append(Path(pattern))

    if not expanded:
        typer.echo("No run directories found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading {len(expanded)} run(s)...")
    results = []
    for d in expanded:
        r = _load_run(d)
        if r is not None:
            results.append(r)
            typer.echo(
                f"  OK  {d}  "
                f"(best epoch {r['best_epoch']}, val_loss={r['val_loss']:.6f})"
            )

    if not results:
        typer.echo("No valid runs found.", err=True)
        raise typer.Exit(1)

    _print_table(results)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    typer.echo(f"Summary written to {output}")


if __name__ == "__main__":
    app()
