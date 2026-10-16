import zipfile
from pathlib import Path

import pytest

from ai_spine_med.assets import extract_zip, sha256_file


def test_zip_traversal_is_rejected_before_extraction(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("safe.txt", "safe")
        zipped.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "output" / "safe.txt").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_zip_extract_and_hash(tmp_path: Path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("scan/image", b"abc")
    extract_zip(archive, tmp_path / "output")
    assert sha256_file(tmp_path / "output/scan/image") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
