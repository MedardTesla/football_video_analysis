"""Point d'entrée du worker."""
from __future__ import annotations

import logging

from football_analysis.config import Config

from .jobs import JobStore
from .notify import from_environment
from .settings import DATA_ROOT
from .storage import Storage


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    from .worker import serve

    notifier = from_environment()
    log = logging.getLogger("worker")
    log.info("worker démarré, données dans %s", DATA_ROOT)
    log.info("notifications : %s", type(notifier).__name__)
    serve(
        JobStore(DATA_ROOT / "jobs.db"),
        Storage(DATA_ROOT / "videos"),
        Config(),
        notifier=notifier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
