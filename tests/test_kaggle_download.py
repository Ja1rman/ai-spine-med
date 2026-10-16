"""Official API protocol tests use fake files and never require credentials."""

import csv
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_spine_med.data import LEVELS, METADATA_FILES, prepare_rsna
from ai_spine_med.kaggle_download import (
    DownloadError,
    _api,
    _file_index,
    download_metadata,
    download_sample,
)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifests(root, out):
    grades, series = [], []
    for index in range(9):
        study = str(100 + index)
        grades.append(
            {
                "study_id": study,
                **{
                    column: ("Normal/Mild", "Moderate", "Severe")[index // 3]
                    for column in LEVELS.values()
                },
            }
        )
        series.append(
            {"study_id": study, "series_id": "10", "series_description": "Sagittal T2/STIR"}
        )
    write_csv(root / "train.csv", ["study_id", *LEVELS.values()], grades)
    write_csv(
        root / "train_series_descriptions.csv",
        ["study_id", "series_id", "series_description"],
        series,
    )
    write_csv(
        root / "train_label_coordinates.csv",
        ["study_id", "series_id", "instance_number", "condition", "level", "x", "y"],
        [],
    )
    prepare_rsna(root, out)


class FakeApi:
    def __init__(self, names=(), *, nested=False, zipped=False):
        self.names = list(names)
        self.nested = nested
        self.zipped = zipped
        self.downloads = []
        self.pages = []

    def competition_list_files(self, competition, page_token=None, page_size=200):
        self.pages.append(page_token)
        # Force multiple pages even for the tiny synthetic cohort.
        middle = len(self.names) // 2
        selected = self.names[:middle] if page_token is None else self.names[middle:]
        return SimpleNamespace(
            files=[SimpleNamespace(name=name) for name in selected],
            next_page_token="next" if page_token is None else None,
        )

    def competition_download_file(self, competition, file_name, path, force, quiet):
        self.downloads.append(file_name)
        target = Path(path) / (file_name if self.nested else Path(file_name).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.zipped:
            with zipfile.ZipFile(str(target) + ".zip", "w") as archive:
                archive.writestr(file_name, b"synthetic official-API fixture")
        else:
            target.write_bytes(b"synthetic official-API fixture")


@pytest.mark.parametrize(
    "nested,zipped", [(False, False), (True, False), (False, True), (True, True)]
)
def test_exact_metadata_only_and_compatible_layouts(tmp_path, nested, zipped):
    api = FakeApi(nested=nested, zipped=zipped)
    report = download_metadata(tmp_path, api=api)
    assert api.downloads == list(METADATA_FILES)
    assert report["downloaded"] == 3
    assert all((tmp_path / name).is_file() for name in METADATA_FILES)
    report = download_metadata(tmp_path, api=api)
    assert report["already_present"] == 3
    assert len(api.downloads) == 3


def test_smoke_download_is_training_only_and_complete_across_pages(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    manifests(root, out)
    names = [
        f"train_images/{study}/10/{instance}.dcm"
        for study in range(100, 109)
        for instance in range(1, 4)
    ]
    api = FakeApi(names, nested=True)
    result = download_sample(root, out, count=3, api=api)
    with (out / "studies.csv").open() as handle:
        train = {row["study_id"] for row in csv.DictReader(handle) if row["split"] == "train"}
    assert result["n_series"] == 3
    assert result["n_files"] == 9
    assert all(item["study_id"] in train for item in result["series"])
    assert all(Path(root / name).is_file() for name in api.downloads)
    assert api.pages == [None, "next"]
    assert len(set(result_item["study_id"] for result_item in result["series"])) == 3
    repeated = download_sample(root, out, count=3, api=api)
    assert repeated["already_present"] == 9
    assert api.pages == [None, "next"]  # Complete index is cached.


def test_file_limit_stops_before_any_image_download(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    manifests(root, out)
    api = FakeApi(
        [
            f"train_images/{study}/10/{instance}.dcm"
            for study in range(100, 109)
            for instance in range(3)
        ]
    )
    with pytest.raises(DownloadError, match="exceed"):
        download_sample(root, out, api=api, max_files=2)
    assert api.downloads == []


def test_changed_split_manifest_cannot_drive_download(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    manifests(root, out)
    path = out / "series.csv"
    path.write_text(path.read_text().replace("validation", "train"))
    api = FakeApi()
    with pytest.raises(DownloadError, match="checksum"):
        download_sample(root, out, api=api)
    assert not api.pages
    assert not api.downloads


def test_repeated_page_token_never_creates_partial_index(tmp_path):
    class BrokenApi:
        def competition_list_files(self, *args, **kwargs):
            return {"files": [{"name": "train.csv"}], "nextPageToken": "same"}

    with pytest.raises(DownloadError, match="repeated"):
        _file_index(BrokenApi(), tmp_path, refresh=False)
    assert not (tmp_path / "_download_cache" / "kaggle_file_index.json").exists()


def test_service_error_does_not_echo_tokens(tmp_path):
    class FailedApi:
        def competition_download_file(self, *args, **kwargs):
            raise RuntimeError("https://signed.example/?token=TOPSECRET")

    with pytest.raises(DownloadError) as error:
        download_metadata(tmp_path, api=FailedApi())
    assert "TOPSECRET" not in str(error.value)
    assert "rules" in str(error.value)


def test_modern_sdk_auth_systemexit_is_silent_and_actionable(monkeypatch, capsys):
    class UnauthenticatedApi:
        def authenticate(self):
            print("Visit auth.example/?authorization=TOPSECRET")
            raise SystemExit(1)

    module = SimpleNamespace(KaggleApi=UnauthenticatedApi)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", module)
    with pytest.raises(DownloadError, match="authentication failed"):
        _api()
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


def test_auth_does_not_intercept_keyboard_interrupt(monkeypatch):
    class InterruptedApi:
        def authenticate(self):
            raise KeyboardInterrupt

    monkeypatch.setitem(
        sys.modules, "kaggle.api.kaggle_api_extended", SimpleNamespace(KaggleApi=InterruptedApi)
    )
    with pytest.raises(KeyboardInterrupt):
        _api()


def test_modern_sdk_file_response_supports_snake_case_fields(tmp_path):
    class SdkApi:
        def competition_list_files(self, competition, page_token, page_size):
            assert page_size == 200
            return SimpleNamespace(files=[SimpleNamespace(name="train.csv")], next_page_token="")

    assert _file_index(SdkApi(), tmp_path, refresh=False) == ["train.csv"]
