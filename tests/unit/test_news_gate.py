from __future__ import annotations

from datetime import UTC, datetime

from slytrade.runtime.news_gate import NewsEvent, NewsGate, load_news_gate


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def test_disabled_gate_never_blocks() -> None:
    gate = NewsGate(enabled=False)
    assert not gate.is_red_folder(_dt("2026-08-14T12:00:00+00:00"))
    assert gate.reason() is None


def test_event_window_blocks_inside_allows_outside() -> None:
    gate = NewsGate(
        enabled=True,
        events=(NewsEvent("NFP", _dt("2026-09-04T12:30:00+00:00"), _dt("2026-09-04T13:45:00+00:00")),),
        quiet_before_minutes=0,
        quiet_after_minutes=0,
    )
    assert gate.is_red_folder(_dt("2026-09-04T13:00:00+00:00"))
    assert not gate.is_red_folder(_dt("2026-09-04T14:00:00+00:00"))
    assert gate.reason(_dt("2026-09-04T13:00:00+00:00")) is not None
    assert "NFP" in (gate.reason(_dt("2026-09-04T13:00:00+00:00")) or "")


def test_quiet_padding_applied() -> None:
    gate = NewsGate(
        enabled=True,
        events=(NewsEvent("CPI", _dt("2026-09-10T12:30:00+00:00"), _dt("2026-09-10T13:30:00+00:00")),),
        quiet_before_minutes=15,
        quiet_after_minutes=15,
    )
    assert gate.is_red_folder(_dt("2026-09-10T12:16:00+00:00"))  # just inside quiet-before
    assert gate.is_red_folder(_dt("2026-09-10T13:44:00+00:00"))  # just inside quiet-after
    assert not gate.is_red_folder(_dt("2026-09-10T12:14:00+00:00"))


def test_recurring_nfp_first_friday() -> None:
    gate = NewsGate(enabled=True, enable_recurring=True, year=2026, quiet_before_minutes=0, quiet_after_minutes=0)
    # 2026-09-04 is the first Friday of September 2026.
    assert gate.is_red_folder(_dt("2026-09-04T12:45:00+00:00"))
    assert not gate.is_red_folder(_dt("2026-09-11T12:45:00+00:00"))


def test_event_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        NewsEvent("bad", _dt("2026-09-04T13:00:00+00:00"), _dt("2026-09-04T12:00:00+00:00"))
    with pytest.raises(ValueError):
        NewsEvent("", _dt("2026-09-04T12:00:00+00:00"), _dt("2026-09-04T13:00:00+00:00"))


def test_load_news_gate_missing_file_disabled(tmp_path) -> None:
    gate = load_news_gate(tmp_path / "missing.yaml")
    assert not gate.enabled


def test_load_news_gate_reads_yaml(tmp_path) -> None:
    path = tmp_path / "news.yaml"
    path.write_text(
        "enabled: true\nquiet_before_minutes: 10\nquiet_after_minutes: 10\n"
        "events:\n  - name: FOMC\n    start: '2026-09-16T18:00:00'\n    end: '2026-09-16T19:30:00'\n",
        encoding="utf-8",
    )
    gate = load_news_gate(path, year=2026)
    assert gate.enabled
    assert gate.is_red_folder(_dt("2026-09-16T18:30:00+00:00"))
