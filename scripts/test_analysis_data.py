from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from pandas.api.types import (
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_categorical_dtype,
)

DATA_PATH = os.path.join("data", "cleaned_movies.csv")


@pytest.fixture(scope="module")
def movies():
    """Load the cleaned movies dataset once for all tests."""
    df = pd.read_csv(DATA_PATH)

    # Normalise column names a bit
    if "Tickets.Sold" in df.columns and "tickets_sold" not in df.columns:
        df["tickets_sold"] = df["Tickets.Sold"]

    if "release_date" in df.columns:
        df["release_date_parsed"] = pd.to_datetime(df["release_date"])
        df["release_month"] = df["release_date_parsed"].dt.month

    return df


def test_schema_and_ranges(movies: pd.DataFrame):
    """Basic data-quality checks on the real dataset."""

    # Required columns
    required = [
        "Year",
        "Genre",
        "Distributor",
        "MPAA",
        "tickets_sold",
    ]
    for col in required:
        assert col in movies.columns, f"Missing required column: {col}"

    # Year and month ranges
    assert movies["Year"].between(2006, 2015).all()

    # Tickets non-negative
    assert (movies["tickets_sold"] >= 0).all()

    # No obvious NA problems in core fields
    assert not movies[required].isna().any().any()

    # MPAA values are from known set
    allowed_mpaa = {"G", "PG", "PG-13", "R", "Not Rated", "NC-17"}
    assert set(movies["MPAA"].dropna().unique()).issubset(allowed_mpaa)


def test_log_transform_and_model_performance(movies: pd.DataFrame):
    """
    Check that the log10(tickets_sold) transformation behaves as expected
    and that a simple linear model achieves performance in a reasonable band.
    """

    # Drop zeros for log10; if many zeros, you can switch to log1p instead.
    movies = movies[movies["tickets_sold"] > 0].copy()
    movies["log10_tickets_sold"] = np.log10(movies["tickets_sold"])

    # Sanity: log10 values finite and within plausible range
    q1 = movies["log10_tickets_sold"].quantile(0.01)
    q99 = movies["log10_tickets_sold"].quantile(0.99)
    assert q1 > 0, "Lower 1% of log10 tickets is suspiciously low"
    assert q99 < 8.5, "Upper 1% of log10 tickets is suspiciously high"

    # Simple model similar to paper spec
    feature_cols = [
        "Year",
        "release_month",
        "Genre",
        "Distributor",
        "MPAA",
    ]
    X = movies[feature_cols].copy()
    y = movies["log10_tickets_sold"].values

    numeric_features = ["Year", "release_month"]
    categorical_features = ["Genre", "Distributor", "MPAA"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", pre),
            ("lm", LinearRegression()),
        ]
    )

    model.fit(X, y)
    y_hat = model.predict(X)
    r2 = r2_score(y, y_hat)

    assert 0.4 < r2 < 0.9, f"Unexpected R^2 on real data: {r2:.3f}"


def test_no_extreme_outliers_in_log_scale(movies: pd.DataFrame):
    """Check there are no bizarre outliers indicating data errors (e.g., extra zeros)."""

    movies = movies[movies["tickets_sold"] > 0].copy()
    movies["log10_tickets_sold"] = np.log10(movies["tickets_sold"])

    q1 = movies["log10_tickets_sold"].quantile(0.25)
    q3 = movies["log10_tickets_sold"].quantile(0.75)
    iqr = q3 - q1

    upper_bound = q3 + 4 * iqr  # pretty generous
    lower_bound = q1 - 4 * iqr

    # Allow a *few* beyond this, but not many
    mask = ~movies["log10_tickets_sold"].between(lower_bound, upper_bound)
    frac_outliers = mask.mean()

    assert frac_outliers < 0.01, "Too many extreme outliers in log tickets: possible data errors."

    from pandas.api.types import (
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_categorical_dtype,
)
def test_column_dtypes(movies: pd.DataFrame):
    """Check that key columns have the expected dtypes."""

    # Year + month should be integer-like
    assert is_integer_dtype(movies["Year"]), "Year should be integer-like"
    assert is_integer_dtype(movies["release_month"]), "release_month should be integer-like"

    # tickets_sold should be numeric (usually int)
    assert is_numeric_dtype(movies["tickets_sold"]), "tickets_sold should be numeric"

    # Categorical text columns: object or categorical
    for col in ["Genre", "Distributor", "MPAA"]:
        assert (
            is_object_dtype(movies[col]) or is_categorical_dtype(movies[col])
        ), f"{col} should be object or categorical"

    # Title indicator columns: bool or 0/1 ints (if present)
    title_flag_cols = [
        c for c in movies.columns
        if c.startswith("title_has_")
    ]
    for col in title_flag_cols:
        assert (
            is_bool_dtype(movies[col]) or is_integer_dtype(movies[col])
        ), f"{col} should be boolean or integer 0/1"
