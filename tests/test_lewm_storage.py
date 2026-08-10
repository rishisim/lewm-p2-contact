import json
from pathlib import Path

import pytest

import lewm_storage


def write_storage(tmp_path: Path, storage_id: str = "lewm-governed-storage-v1") -> Path:
    root = tmp_path / "lewm-storage"
    (root / "stable-worldmodel").mkdir(parents=True)
    (root / ".lewm-storage.json").write_text(
        json.dumps(
            {
                "storage_id": storage_id,
                "volume_uuid": "C4E3E4D2-BC24-45D8-9BD0-F7F477F1EC90",
            }
        )
    )
    (root / "manifest.json").write_text("{}")
    return root


def test_override_resolves_and_exports_stablewm_home(tmp_path, monkeypatch):
    root = write_storage(tmp_path)
    monkeypatch.setenv("LEWM_STORAGE_ROOT", str(root))

    assert lewm_storage.require_external_storage() == root / "stable-worldmodel"
    assert lewm_storage.os.environ["STABLEWM_HOME"] == str(root / "stable-worldmodel")


def test_missing_storage_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_STORAGE_ROOT", str(tmp_path / "missing"))

    with pytest.raises(lewm_storage.ExternalStorageUnavailable, match="Connect the USB"):
        lewm_storage.require_external_storage()


def test_wrong_marker_is_rejected(tmp_path, monkeypatch):
    root = write_storage(tmp_path, storage_id="wrong")
    monkeypatch.setenv("LEWM_STORAGE_ROOT", str(root))

    with pytest.raises(lewm_storage.ExternalStorageUnavailable, match="unrecognized"):
        lewm_storage.require_external_storage()
