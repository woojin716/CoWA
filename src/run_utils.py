"""Shared utilities for experiment runners."""
import datetime
from pathlib import Path


def make_run_dir(base: str = 'results') -> Path:
    """Create and return a fresh run directory: <base>/<YYYYMMDD>/<NNN>/.

    NNN is the next zero-padded 3-digit index (000, 001, ...) under today's
    date directory. The new directory is created on disk before returning.
    """
    date_dir = Path(base) / datetime.datetime.now().strftime('%Y%m%d')
    date_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p.name for p in date_dir.iterdir()
        if p.is_dir() and len(p.name) == 3 and p.name.isdigit()
    )
    next_idx = int(existing[-1]) + 1 if existing else 0

    run_dir = date_dir / f"{next_idx:03d}"
    run_dir.mkdir()
    return run_dir


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else 'results'
    print(make_run_dir(base))
