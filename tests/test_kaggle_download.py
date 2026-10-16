"""Official API protocol tests use fake files and never require credentials."""

import csv
import json
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_spine_med.data import LEVELS, METADATA_FILES, prepare_rsna
from ai_spine_med.kaggle_download import (
    DownloadError,
    _api,
    _file_index,
    _series_file_index,
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


@pytest.mark.parametrize("operation", ["metadata", "index"])
def test_rules_acceptance_error_is_specific_without_echoing_private_details(tmp_path, operation):
    class HTTPError(Exception):
        response = SimpleNamespace(
            status_code=403,
            json=lambda: {
                "error": {
                    "message": "PRIVATE_SERVER_MESSAGE",
                    "details": [
                        {
                            "reason": "RulesAcceptanceRequired",
                            "metadata": {"signed_url": "https://private.example/?token=TOPSECRET"},
                        }
                    ],
                }
            },
        )

    class FailedApi:
        def competition_download_file(self, *args, **kwargs):
            raise HTTPError("PRIVATE_EXCEPTION_TEXT")

        def competition_list_files(self, *args, **kwargs):
            raise HTTPError("PRIVATE_EXCEPTION_TEXT")

    with pytest.raises(DownloadError) as error:
        if operation == "metadata":
            download_metadata(tmp_path, api=FailedApi())
        else:
            _file_index(FailedApi(), tmp_path, refresh=True)
    message = str(error.value)
    assert "not recorded acceptance" in message
    assert "rsna-2024-lumbar-spine-degenerative-classification/rules" in message
    assert "same account" in message
    assert "TOPSECRET" not in message
    assert "PRIVATE_" not in message


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


def test_sdk_directory_listing_selects_only_requested_series_and_handles_pages(
    tmp_path, monkeypatch
):
    import ai_spine_med.kaggle_download as downloader

    requests = []

    class Service:
        def list_data_tree_files(self, request):
            requests.append(request)
            return SimpleNamespace(
                files=[SimpleNamespace(name="2.dcm" if request.page_token else "1.dcm")],
                directories=[],
                total_files=2,
                next_page_token="" if request.page_token else "next",
            )

    class DirectoryApi:
        def build_kaggle_client(self):
            return nullcontext(
                SimpleNamespace(competitions=SimpleNamespace(competition_api_client=Service()))
            )

        def competition_list_files(self, *args, **kwargs):
            pytest.fail("The current SDK must not scan the complete competition index.")

    monkeypatch.setattr(
        downloader,
        "_make_tree_request",
        lambda directory, token: SimpleNamespace(path=directory, page_token=token or ""),
    )
    selected = [("100", "10"), ("101", "20")]
    names = _series_file_index(DirectoryApi(), tmp_path, selected, refresh=False)
    assert names == [
        "train_images/100/10/1.dcm",
        "train_images/100/10/2.dcm",
        "train_images/101/20/1.dcm",
        "train_images/101/20/2.dcm",
    ]
    assert [request.path for request in requests] == ["train_images/100/10"] * 2 + [
        "train_images/101/20"
    ] * 2
    assert _series_file_index(DirectoryApi(), tmp_path, selected, refresh=False) == names
    assert len(requests) == 4


def test_sdk_directory_listing_rejects_incomplete_series_before_downloading(tmp_path, monkeypatch):
    import ai_spine_med.kaggle_download as downloader

    class Service:
        def list_data_tree_files(self, request):
            return SimpleNamespace(
                files=[SimpleNamespace(name="1.dcm")],
                directories=[],
                total_files=18,
                next_page_token="",
            )

    class DirectoryApi:
        def build_kaggle_client(self):
            return nullcontext(
                SimpleNamespace(competitions=SimpleNamespace(competition_api_client=Service()))
            )

    monkeypatch.setattr(
        downloader, "_make_tree_request", lambda directory, token: SimpleNamespace(path=directory)
    )
    with pytest.raises(DownloadError, match="incomplete"):
        _series_file_index(DirectoryApi(), tmp_path, [("100", "10")], refresh=False)
    assert not (tmp_path / "_download_cache" / "series_100_10.json").exists()


@pytest.mark.parametrize(
    "corruption", ["truncated", "missing_total", "boolean_total", "zero_total"]
)
def test_truncated_cache_forces_directory_relisting(tmp_path, monkeypatch, corruption):
    import ai_spine_med.kaggle_download as downloader

    calls = []

    class Service:
        def list_data_tree_files(self, request):
            calls.append(request.path)
            return SimpleNamespace(
                files=[SimpleNamespace(name="1.dcm"), SimpleNamespace(name="2.dcm")],
                directories=[],
                total_files=2,
                next_page_token="",
            )

    class DirectoryApi:
        def build_kaggle_client(self):
            return nullcontext(
                SimpleNamespace(competitions=SimpleNamespace(competition_api_client=Service()))
            )

    monkeypatch.setattr(
        downloader,
        "_make_tree_request",
        lambda directory, token: SimpleNamespace(path=directory),
    )
    selected = [("100", "10")]
    expected = _series_file_index(DirectoryApi(), tmp_path, selected, refresh=False)
    path = tmp_path / "_download_cache" / "series_100_10.json"
    cached = json.loads(path.read_text())
    assert cached["total_files"] == 2
    if corruption == "truncated":
        cached["files"] = cached["files"][:1]
    elif corruption == "missing_total":
        del cached["total_files"]
    elif corruption == "boolean_total":
        cached["total_files"] = True
    else:
        cached["total_files"] = 0
    path.write_text(json.dumps(cached))
    assert _series_file_index(DirectoryApi(), tmp_path, selected, refresh=False) == expected
    assert len(calls) == 2
    assert json.loads(path.read_text())["total_files"] == 2
