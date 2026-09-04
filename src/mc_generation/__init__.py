"""Self-contained MCsquare generation for adota (migrated from datagenerator).

Vendors the minimal MCsquare-running path (PlanPencil + config writers, single-
beamlet runner) with no nvidia.dali / totalsegmentator dependency. The engine
install lives at an independent home (see docs/mcsquare_engine.md); working dirs
are on /scratch, never in the repo.
"""
