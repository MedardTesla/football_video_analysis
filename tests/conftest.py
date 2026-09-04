"""Fixtures partagées entre les suites de test du service."""
from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API montée sur un stockage jetable.

    `settings` est rechargé avant `api` : c'est lui qui lit FA_DATA_ROOT, et
    ne recharger qu'api ferait partager la même base à tous les tests.
    """
    monkeypatch.setenv("FA_DATA_ROOT", str(tmp_path))
    import importlib

    from fastapi.testclient import TestClient

    import service.api as api
    import service.settings as settings

    importlib.reload(settings)
    importlib.reload(api)
    return TestClient(api.app), api
