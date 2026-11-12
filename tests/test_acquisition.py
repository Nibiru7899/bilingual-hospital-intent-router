from __future__ import annotations

import pytest

from ml.acquisition import COMMIT, acquire
from ml.common import sha256, write_json


def test_source_manifest_revalidation_and_tampering(tmp_path, monkeypatch):
    source = tmp_path / "mediq_full.json"
    source.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr("ml.acquisition.DATA_SHA256", sha256(source))
    write_json(tmp_path / "source_manifest.json", {"commit": COMMIT, "files": [{"name": source.name, "sha256": sha256(source)}]})
    assert acquire(tmp_path)["commit"] == COMMIT
    source.write_text("changed fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        acquire(tmp_path)


def test_download_limit_does_not_create_final_file(tmp_path, monkeypatch):
    import io
    from ml.acquisition import download
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: io.BytesIO(b"fixture data too large"))
    target = tmp_path / "dataset.json"
    with pytest.raises(ValueError, match="resource limit"):
        download("https://example.invalid/fixture", target, limit=2)
    assert not target.exists()


def test_pinned_checksum_cannot_be_overridden_by_manifest(tmp_path):
    source = tmp_path / "mediq_full.json"
    source.write_text("not MEDIQ", encoding="utf-8")
    write_json(tmp_path / "source_manifest.json", {"commit": COMMIT, "files": [{"name": source.name, "sha256": sha256(source)}]})
    with pytest.raises(ValueError, match="pinned release"):
        acquire(tmp_path)
