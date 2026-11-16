import json
import tarfile

import pytest

from copuzzle_baselines.study.packaging import build_archives, notice_members, verify_archives
from copuzzle_baselines.study.storage import file_hash


def test_archives_are_deterministic_complete_and_chunked(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"first"); b.write_bytes(b"second")
    files = {"root/b": b, "root/a": a}
    first = build_archives(files, tmp_path / "first", identity={"role": "test"}, part_bytes=6)
    second = build_archives(files, tmp_path / "second", identity={"role": "test"}, part_bytes=6)
    assert first == second and first["archives"] == 2 and first["members"] == 2
    assert (tmp_path / "first/part-001.tar.gz").read_bytes() == (tmp_path / "second/part-001.tar.gz").read_bytes()
    with pytest.raises(FileExistsError):
        build_archives(files, tmp_path / "first", identity={})
    with pytest.raises(ValueError, match="exceeds"):
        build_archives(files, tmp_path / "too-small", identity={}, part_bytes=4)


def test_archive_validation_rejects_corruption_and_missing_members(tmp_path):
    source = tmp_path / "a"; source.write_bytes(b"content")
    output = tmp_path / "archive"
    build_archives({"a": source}, output, identity={})
    path = output / "part-001.tar.gz"
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_archives(output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["archives"] = []
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits"):
        verify_archives(output)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\b", "a//b", ".", "C:/escape"])
def test_archive_paths_cannot_escape_or_alias(tmp_path, name):
    source = tmp_path / "source"; source.write_bytes(b"a")
    with pytest.raises(ValueError, match="relative"):
        build_archives({name: source}, tmp_path / "out", identity={})


def test_archive_links_are_rejected_even_with_updated_outer_checksum(tmp_path):
    source = tmp_path / "source"; source.write_bytes(b"a")
    output = tmp_path / "out"
    build_archives({"safe": source}, output, identity={})
    path = output / "part-001.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("safe"); entry.type = tarfile.SYMTYPE; entry.linkname = "/etc/passwd"
        archive.addfile(entry)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["archives"][0].update(sha256=file_hash(path), bytes=path.stat().st_size)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonregular"):
        verify_archives(output)


def test_notices_are_bound_outside_the_frozen_artifact_tree(tmp_path):
    notice = tmp_path / "CODE_LICENSE.proposed.txt"
    notice.write_text('Test proposal, not an adopted license.\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "copuzzle.distribution_notices.v1", "state": "proposal",
                                    "files": {notice.name: file_hash(notice)}}))
    files = {"data/inventory.json": manifest}
    before = dict(files)
    result = notice_members(files, manifest)
    assert result["state"] == "proposal"
    assert all(files[k] == v for k, v in before.items())
    assert set(files)-set(before) == {"notices/manifest.json", "notices/CODE_LICENSE.proposed.txt"}
    notice.write_text('changed')
    with pytest.raises(ValueError, match="checksum"):
        notice_members({}, manifest)
