from pathlib import Path

import pytest

from pathox import WSIReader


def test_missing_wsi_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        WSIReader("/tmp/pathox_missing_slide.svs")


def test_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        WSIReader(tmp_path)


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    fake_slide = tmp_path / "fake.svs"
    fake_slide.write_text("not a real whole-slide image")

    with pytest.raises(ValueError, match="Unsupported"):
        WSIReader(fake_slide)
