from datetime import UTC, datetime

from slytrade.features.sessions import session_label, session_one_hot


def test_session_labels():
    assert session_label(datetime(2026, 1, 1, 1, tzinfo=UTC)) == "asia"
    assert session_label(datetime(2026, 1, 1, 8, tzinfo=UTC)) == "london"
    assert session_label(datetime(2026, 1, 1, 13, tzinfo=UTC)) == "ny_am"
    assert session_label(datetime(2026, 1, 1, 18, tzinfo=UTC)) == "ny_pm"
    assert session_label(datetime(2026, 1, 1, 22, tzinfo=UTC)) == "other"


def test_session_one_hot_has_one_active_label():
    encoded = session_one_hot(datetime(2026, 1, 1, 8, tzinfo=UTC))

    assert encoded["session_london"] == 1.0
    assert sum(encoded.values()) == 1.0
