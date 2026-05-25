"""
SmartCleaner - tests/test_config_and_limits.py
------------------------------------------------
Tests for environment variable loading and file size limits.
"""

import io
import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api import app

client = TestClient(app)


def make_csv(rows: int = 5) -> bytes:
    df = pd.DataFrame({
        "name":   [f"Person{i}" for i in range(rows)],
        "age":    [25.0 + i for i in range(rows)],
        "salary": [50000.0 + i * 1000 for i in range(rows)],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def make_oversized_file(size_mb: float) -> bytes:
    size_bytes = int(size_mb * 1024 * 1024)
    return b"x" * size_bytes


# ── Config Tests ──────────────────────────────────────────────

class TestSettings:

    def test_settings_object_exists(self):
        from src.config import settings
        assert settings is not None

    def test_max_file_size_mb_is_positive(self):
        from src.config import settings
        assert settings.MAX_FILE_SIZE_MB > 0
        assert isinstance(settings.MAX_FILE_SIZE_MB, int)

    def test_max_file_size_bytes_is_correct(self):
        from src.config import settings
        expected = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        assert settings.MAX_FILE_SIZE_BYTES == expected

    def test_knn_neighbors_is_positive(self):
        from src.config import settings
        assert settings.KNN_NEIGHBORS >= 1

    def test_outlier_contamination_is_valid_range(self):
        from src.config import settings
        assert 0.0 < settings.OUTLIER_CONTAMINATION <= 0.5

    def test_environment_is_string(self):
        from src.config import settings
        assert isinstance(settings.ENVIRONMENT, str)

    def test_secret_key_exists(self):
        from src.config import settings
        assert settings.SECRET_KEY is not None
        assert len(settings.SECRET_KEY) > 0

    def test_schema_path_is_string(self):
        from src.config import settings
        assert isinstance(settings.SCHEMA_PATH, str)

    def test_default_values_work_without_env_file(self):
        from src.config import Settings
        s = Settings()
        assert s.MAX_FILE_SIZE_MB > 0
        assert s.KNN_NEIGHBORS   > 0
        assert s.API_PORT        > 0


# ── File Size Limit Tests ─────────────────────────────────────

class TestFileSizeLimit:

    def test_normal_file_is_accepted(self):
        response = client.post(
            "/clean",
            files={"file": ("data.csv", make_csv(10), "text/csv")},
        )
        assert response.status_code == 200

    def test_oversized_file_returns_413(self):
        from src.config import settings
        oversized = make_oversized_file(settings.MAX_FILE_SIZE_MB + 1)
        response = client.post(
            "/clean",
            files={"file": ("huge.csv", oversized, "text/csv")},
        )
        assert response.status_code == 413

    def test_413_error_message_mentions_file_size(self):
        from src.config import settings
        oversized = make_oversized_file(settings.MAX_FILE_SIZE_MB + 1)
        response = client.post(
            "/clean",
            files={"file": ("huge.csv", oversized, "text/csv")},
        )
        detail = response.json().get("detail", "")
        assert "MB" in detail or "large" in detail.lower()

    def test_oversized_file_rejected_on_inspect_endpoint(self):
        from src.config import settings
        oversized = make_oversized_file(settings.MAX_FILE_SIZE_MB + 1)
        response = client.post(
            "/inspect",
            files={"file": ("huge.csv", oversized, "text/csv")},
        )
        assert response.status_code == 413

    def test_oversized_file_rejected_on_schema_save_endpoint(self):
        from src.config import settings
        oversized = make_oversized_file(settings.MAX_FILE_SIZE_MB + 1)
        response = client.post(
            "/schema/save",
            files={"file": ("huge.csv", oversized, "text/csv")},
        )
        assert response.status_code == 413

    def test_small_file_is_accepted(self):
        from src.config import settings
        small_csv = make_csv(5)
        assert len(small_csv) < settings.MAX_FILE_SIZE_BYTES
        response = client.post(
            "/clean",
            files={"file": ("data.csv", small_csv, "text/csv")},
        )
        assert response.status_code == 200

    def test_health_endpoint_shows_max_file_size(self):
        from src.config import settings
        response = client.get("/health")
        data = response.json()
        assert "max_file_size_mb" in data
        assert data["max_file_size_mb"] == settings.MAX_FILE_SIZE_MB