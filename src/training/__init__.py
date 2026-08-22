"""Training-loop infrastructure for ADoTA.

``scripts/train_adota.py`` is the entry point; this package holds the pieces it
wires together.

Modules:
- ``data``, ``factory``: dataset/dataloader construction, model and optimizer.
- ``loop``, ``losses``: the training step and its objectives.
- ``validation``, ``binning``, ``attention``, ``gpr_pool``: per-epoch evaluation.
- ``logging_utils``, ``run_dir``, ``checkpoints``, ``diagnostics``: run
  scaffolding (formatted logs, run directory and manifest, checkpoint retention
  and resume, shutdown handling and numerical diagnostics).
- ``utils``: small shared helpers.
"""
