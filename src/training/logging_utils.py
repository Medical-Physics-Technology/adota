"""Log formatting and structured log output for a training run.

Flow:
1. :func:`setup_training_logging` installs :class:`RelativeTimeFormatter` on
   both the run-directory log file and the console, so every line carries a
   phase tag and the time elapsed since the run started.
2. :func:`log_phase` / :func:`log_banner` / :func:`log_section` emit ASCII-only
   structured output at consistent widths.
3. :func:`silence_pymedphys` mutes pymedphys's per-call gamma logging, which
   would otherwise add five INFO lines per validation sample.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Logging: phase-tagged formatter ─────────────────────────────────────────


BANNER_WIDTH = 80
PHASE_FIELD_WIDTH = 5  # widest tag we use (TRAIN, EPOCH, ERROR, ...)
_DEFAULT_PHASE = "LOG"


class RelativeTimeFormatter(logging.Formatter):
    """Format log records as ``HH:MM:SS [PHASE] message``.

    The time component is the elapsed wall time since training started,
    not an absolute timestamp -- the run directory name already records
    the start date, and relative timing makes a long training log much
    easier to skim.

    The phase tag is taken from ``record.phase`` if the call site passed
    ``extra={"phase": "TRAIN"}``; otherwise it falls back to
    :data:`_DEFAULT_PHASE`.

    Args:
        start_time: ``time.time()`` value captured at run start.
        phase_width: Width to left-justify the phase tag inside the
            brackets (defaults to :data:`PHASE_FIELD_WIDTH`).
    """

    def __init__(self, start_time: float, phase_width: int = PHASE_FIELD_WIDTH):
        super().__init__()
        self.start_time = start_time
        self.phase_width = phase_width

    def format(self, record: logging.LogRecord) -> str:
        elapsed = max(0.0, record.created - self.start_time)
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        phase = getattr(record, "phase", _DEFAULT_PHASE)
        return f"{time_str} [{phase:<{self.phase_width}}] {record.getMessage()}"


def setup_training_logging(
    run_dir: Path,
    start_time: float,
    verbose: bool = False,
    log_filename: str = "training.log",
) -> Path:
    """Install :class:`RelativeTimeFormatter` on the root logger.

    Wipes any handlers configured by an earlier ``setup_logging`` call so
    the training script gets a clean stream into both stdout and the
    run-dir log file. Returns the log-file path.
    """
    log_file = run_dir / log_filename
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = RelativeTimeFormatter(start_time=start_time)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_file


def silence_pymedphys() -> None:
    """Suppress pymedphys's INFO-level gamma logging.

    pymedphys emits five INFO lines per ``gamma(...)`` call describing
    its normalisation choices; for a 20-sample GPR subset that's 100
    lines we don't want in the training log. We push its whole logger
    tree to WARNING.
    """
    for name in ("pymedphys", "pymedphys._gamma", "pymedphys.gamma"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Logging: structured helpers ─────────────────────────────────────────────


def log_phase(
    phase: str,
    message: str,
    *,
    level: int = logging.INFO,
    target_logger: Optional[logging.Logger] = None,
) -> None:
    """Emit one log line tagged with ``[phase]``.

    Args:
        phase: Short tag (e.g. ``"TRAIN"``, ``"GPR"``). Kept under
            :data:`PHASE_FIELD_WIDTH` characters for column alignment.
        message: The text.
        level: Logging level (default INFO).
        target_logger: Logger to emit through. Defaults to the root
            logger so any module can call this helper.
    """
    target = target_logger if target_logger is not None else logging.getLogger()
    target.log(level, message, extra={"phase": phase})


def log_banner(title: str, *, char: str = "=") -> None:
    """Emit an 80-char banner with a centered title.

    Three lines: a separator, the title (centered), another separator.
    Banner lines are raw (no time / phase prefix) so they stand out.
    """
    root = logging.getLogger()
    # We bypass the formatter for banners so the separators are full-width.
    for handler in root.handlers:
        handler.stream.write(char * BANNER_WIDTH + "\n")
        handler.stream.write(title.center(BANNER_WIDTH) + "\n")
        handler.stream.write(char * BANNER_WIDTH + "\n")
        handler.stream.flush()


def log_section(title: str, *, char: str = "=") -> None:
    """Emit a section separator with the title inline.

    Two lines: an 80-char rule above and below a single line containing
    the title (indented two spaces).
    """
    root = logging.getLogger()
    rule = char * BANNER_WIDTH
    for handler in root.handlers:
        handler.stream.write(rule + "\n")
        handler.stream.write(f"  {title}\n")
        handler.stream.write(rule + "\n")
        handler.stream.flush()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {sec:02d}s"
