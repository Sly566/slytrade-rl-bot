from datetime import UTC, datetime
from io import BytesIO
from zipfile import ZipFile

from slytrade.data.exness_archive import (
    ExnessArchiveDownloader,
    build_exness_month_url,
    iter_month_starts,
    normalize_exness_symbol,
    normalize_exness_tick_csv,
)


def make_zip_bytes(csv_name: str = "Exness_XAUUSD_2026_07.csv") -> bytes:
    csv_text = "Timestamp,Bid,Ask\n2026-07-01 00:00:00.123,2400.10,2400.34\n2026-07-01 00:00:01.456,2400.20,2400.44\n"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(csv_name, csv_text)
    return buffer.getvalue()


def test_normalize_exness_symbol_strips_common_suffix():
    assert normalize_exness_symbol("xauusdm") == "XAUUSD"
    assert normalize_exness_symbol("EURUSD") == "EURUSD"


def test_build_exness_month_url():
    assert build_exness_month_url("XAUUSD", 2026, 7) == "https://ticks.ex2archive.com/ticks/XAUUSD/2026/07/Exness_XAUUSD_2026_07.zip"


def test_iter_month_starts():
    months = iter_month_starts(datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC))

    assert [month.month for month in months] == [1, 2, 3]


def test_normalize_exness_tick_csv():
    frame = normalize_exness_tick_csv(
        b"Timestamp,Bid,Ask\n2026-07-01 00:00:00.123,2400.10,2400.34\n",
        "XAUUSD",
    )

    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "XAUUSD"
    assert round(float(frame.loc[0, "spread"]), 2) == 0.24
    assert round(float(frame.loc[0, "mid"]), 2) == 2400.22


def test_downloader_collects_mock_month(tmp_path, monkeypatch):
    downloader = ExnessArchiveDownloader(tmp_path)

    def fake_download(url: str, *, timeout: int = 60) -> bytes:
        assert "Exness_XAUUSD_2026_07.zip" in url
        return make_zip_bytes()

    monkeypatch.setattr(downloader, "_download_zip_bytes", fake_download)
    result = downloader.collect(
        "XAUUSDm",
        datetime(2026, 7, 1, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert result.symbol == "XAUUSD"
    assert result.rows == 2
    assert result.file_count == 1
    assert result.months_attempted == 1
    assert result.files[0].path.exists()


def test_downloader_continue_on_error(tmp_path, monkeypatch):
    downloader = ExnessArchiveDownloader(tmp_path)

    def fake_download(url: str, *, timeout: int = 60) -> bytes:
        raise RuntimeError("network down")

    monkeypatch.setattr(downloader, "_download_zip_bytes", fake_download)
    result = downloader.collect(
        "XAUUSD",
        datetime(2026, 7, 1, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
        continue_on_error=True,
    )

    assert result.rows == 0
    assert result.failed_months == 1
    assert result.errors
