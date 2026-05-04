"""
tests/conftest.py
------------------
Shared pytest fixtures for SmartCleaner tests.

A fixture is a reusable piece of test setup. Instead of building
the same DataFrame in 10 different tests, define it once here and
pytest injects it automatically by matching the parameter name.

Usage in a test:
    def test_something(df_dirty_columns):
        result = clean_column_names(df_dirty_columns)
        assert "first_name" in result.columns
"""

import pandas as pd
import pytest


@pytest.fixture
def df_dirty_columns():
    """DataFrame with messy column headers: spaces, mixed case, padding."""
    return pd.DataFrame({
        "  First Name  ": ["Alice", "Bob"],
        "SALARY":         [70000,   80000],
        "Email Address":  ["a@test.com", "b@test.com"],
    })


@pytest.fixture
def df_with_empty_column():
    """DataFrame containing one fully-NaN column."""
    return pd.DataFrame({
        "name":        ["Alice", "Bob"],
        "empty_notes": [None, None],
    })


@pytest.fixture
def df_with_duplicates():
    """DataFrame where one row appears twice."""
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Alice"],
        "salary": [70000,   80000,  70000],
    })


@pytest.fixture
def df_with_messy_text():
    """DataFrame with padded, mixed-case string values."""
    return pd.DataFrame({
        "name":       ["  alice  ", "BOB", "  Carol  "],
        "department": ["ENGINEERING", "marketing", "  Sales  "],
        "salary":     [70000, 80000, 65000],   # numeric — must not be touched
    })


@pytest.fixture
def df_with_missing_values():
    """DataFrame with NaN in both text and numeric columns."""
    return pd.DataFrame({
        "name":   ["Alice", None, "Carol"],
        "salary": [70000.0, None, 65000.0],
    })


@pytest.fixture
def df_full_mess():
    """
    Kitchen-sink fixture: combines every type of problem at once.
    Used for full pipeline (clean_dataset) integration tests.
    """
    return pd.DataFrame({
        "  First Name  ": ["  alice  ", "BOB",     "  alice  ", None   ],
        "  SALARY  ":     [50000.0,     None,       50000.0,    75000.0],
        "Empty Col":      [None,         None,       None,       None   ],
    })