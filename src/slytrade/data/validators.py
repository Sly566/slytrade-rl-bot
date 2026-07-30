from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Severity = Literal["info", "warning", "error"]
DuplicatePolicy = Literal["drop", "keep", "error"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    count: int = 0


@dataclass(frozen=True)
class ValidationReport:
    rows_before: int
    rows_after: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def _required_columns(df: pd.DataFrame, required: list[str], issues: list[ValidationIssue]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        issues.append(ValidationIssue("error", "missing_columns", f"Missing required columns: {missing}", len(missing)))


def validate_tick_frame(
    df: pd.DataFrame,
    *,
    reject_negative_prices: bool = True,
    reject_crossed_spread: bool = True,
    duplicate_policy: DuplicatePolicy = "drop",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a canonical tick DataFrame."""
    rows_before = len(df)
    issues: list[ValidationIssue] = []
    required = ["time", "time_msc", "symbol", "bid", "ask", "spread", "mid"]
    _required_columns(df, required, issues)
    if any(issue.severity == "error" for issue in issues):
        return df.copy(), ValidationReport(rows_before, len(df), issues)

    clean = df.copy()
    duplicate_count = int(clean.duplicated(subset=["time_msc", "bid", "ask", "last"]).sum())
    if duplicate_count:
        issues.append(ValidationIssue("warning", "duplicate_ticks", "Duplicate ticks detected", duplicate_count))
        if duplicate_policy == "drop":
            clean = clean.drop_duplicates(subset=["time_msc", "bid", "ask", "last"])
        elif duplicate_policy == "error":
            issues.append(ValidationIssue("error", "duplicate_ticks_error", "Duplicate ticks not allowed", duplicate_count))

    bad_prices = int(((clean["bid"] <= 0) | (clean["ask"] <= 0)).sum())
    if bad_prices:
        severity: Severity = "error" if reject_negative_prices else "warning"
        issues.append(ValidationIssue(severity, "bad_tick_prices", "Bid/ask must be positive", bad_prices))
        if reject_negative_prices:
            clean = clean[(clean["bid"] > 0) & (clean["ask"] > 0)]

    crossed = int((clean["ask"] < clean["bid"]).sum())
    if crossed:
        severity = "error" if reject_crossed_spread else "warning"
        issues.append(ValidationIssue(severity, "crossed_spread", "Ask is below bid", crossed))
        if reject_crossed_spread:
            clean = clean[clean["ask"] >= clean["bid"]]

    clean = clean.sort_values("time_msc").reset_index(drop=True)
    return clean, ValidationReport(rows_before, len(clean), issues)


def validate_bar_frame(
    df: pd.DataFrame,
    *,
    reject_negative_prices: bool = True,
    duplicate_policy: DuplicatePolicy = "drop",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a canonical bar DataFrame."""
    rows_before = len(df)
    issues: list[ValidationIssue] = []
    required = ["time", "symbol", "timeframe", "open", "high", "low", "close"]
    _required_columns(df, required, issues)
    if any(issue.severity == "error" for issue in issues):
        return df.copy(), ValidationReport(rows_before, len(df), issues)

    clean = df.copy()
    duplicate_count = int(clean.duplicated(subset=["time", "symbol", "timeframe"]).sum())
    if duplicate_count:
        issues.append(ValidationIssue("warning", "duplicate_bars", "Duplicate bars detected", duplicate_count))
        if duplicate_policy == "drop":
            clean = clean.drop_duplicates(subset=["time", "symbol", "timeframe"])
        elif duplicate_policy == "error":
            issues.append(ValidationIssue("error", "duplicate_bars_error", "Duplicate bars not allowed", duplicate_count))

    price_columns = ["open", "high", "low", "close"]
    bad_prices = int((clean[price_columns] <= 0).any(axis=1).sum())
    if bad_prices:
        severity: Severity = "error" if reject_negative_prices else "warning"
        issues.append(ValidationIssue(severity, "bad_bar_prices", "OHLC prices must be positive", bad_prices))
        if reject_negative_prices:
            clean = clean[(clean[price_columns] > 0).all(axis=1)]

    invalid_ranges = int(((clean["high"] < clean["low"]) | (clean["high"] < clean[["open", "close"]].max(axis=1)) | (clean["low"] > clean[["open", "close"]].min(axis=1))).sum())
    if invalid_ranges:
        issues.append(ValidationIssue("error", "invalid_ohlc_range", "OHLC range is inconsistent", invalid_ranges))
        clean = clean[(clean["high"] >= clean["low"]) & (clean["high"] >= clean[["open", "close"]].max(axis=1)) & (clean["low"] <= clean[["open", "close"]].min(axis=1))]

    clean = clean.sort_values("time").reset_index(drop=True)
    return clean, ValidationReport(rows_before, len(clean), issues)
