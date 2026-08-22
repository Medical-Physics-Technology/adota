"""Training-loop infrastructure for ADoTA.

``scripts/train_adota.py`` is the entry point; this package holds the pieces it
wires together.

Modules:
- ``data``, ``factory``: dataset/dataloader construction, model and optimizer.
- ``loop``, ``losses``: the training step and its objectives.
- ``validation``, ``gpr_pool``: per-epoch evaluation and gamma pass rate.
- ``run``: run scaffolding (logging, run directory, checkpoints, diagnostics).
- ``utils``: small shared helpers.
"""
