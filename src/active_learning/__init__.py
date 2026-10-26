"""The active-learning loop: which beamlets are worth simulating next.

- :mod:`.pool`: the CT pool and the validation CTs, with the leakage rule that
  separates them and the registry CSV that records the split.
- :mod:`.candidates`: version-0 candidate generation, and scoring a pool of CTs
  with the input-only difficulty score before anything is simulated.
- :mod:`.sampling`: the strategies (random, score, score_topk, stratified_score).
- :mod:`.oracle`: buying labels, through the Monte Carlo generator.
- :mod:`.dataset`: training on what was bought, unioned with the reference HDF5 set.
- :mod:`.validation`: the frozen difficulty-balanced yardstick, and dR80.
- :mod:`.training`: the retraining step of a cycle.
- :mod:`.loop`: the cycle, its manifest and its resume.
"""
from src.active_learning.candidates import CandidateConfig, generate_candidates, score_pool
from src.active_learning.loop import LoopConfig, run_cycle, run_loop
from src.active_learning.pool import PoolEntry, build_pool, read_pool, write_pool
from src.active_learning.sampling import STRATEGIES, select

__all__ = ["CandidateConfig", "generate_candidates", "score_pool", "LoopConfig",
           "run_cycle", "run_loop", "PoolEntry", "build_pool", "read_pool", "write_pool",
           "STRATEGIES", "select"]
