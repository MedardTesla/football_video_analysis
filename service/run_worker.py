"""Point d'entrée du worker."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from football_analysis.config import Config

from .jobs import JobStore
from .storage import Storage


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    root = Path(os.environ.get("FA_DATA_ROOT", "data/service"))
    from .worker import serve

    logging.getLogger("worker").info("worker démarré, données dans %s", root)
    serve(JobStore(root / "jobs.db"), Storage(root / "videos"), Config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
