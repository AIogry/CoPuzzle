"""Validated public task structure, without transition or planning methods.

Loading this format only reads and validates data. ``rule_from_structure`` is
an explicit preparation/evaluation helper and imports the rule class lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import operator
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING
import zipfile

import numpy as np

if TYPE_CHECKING:
    from .rules import PuzzleRule


STRUCTURE_SCHEMA = "copuzzle.structure.v1"
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024
_SEMANTICS = {
    "rule": "binary-undirected-self-toggle-v1",
    "button_order": "array-index",
    "grid_order": "row-major",
    "grid_coordinate_axes": ["row", "column"],
    "graph_adjacency_axes": ["affected-button", "pressed-button"],
    "graph_adjacency_diagonal": "zero-self-toggle-implicit",
}


def _immutable(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _logical_rule_id(adjacency: np.ndarray) -> str:
    """Hash an already validated ordered binary graph using the legacy format."""
    digest = hashlib.sha256()
    digest.update(b"copuzzle/undirected-self-toggle/v1\0")
    digest.update(str(adjacency.shape[0]).encode("ascii") + b"\0")
    digest.update(np.asarray(adjacency, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _grid_shape(value: object) -> tuple[int, int]:
    try:
        parts = tuple(value)
    except TypeError as error:
        raise ValueError("grid_shape must contain two positive integers") from error
    if len(parts) != 2:
        raise ValueError("grid_shape must contain two positive integers")
    result = []
    for part in parts:
        if isinstance(part, (bool, np.bool_)):
            raise ValueError("grid_shape must contain two positive integers")
        try:
            number = operator.index(part)
        except TypeError as error:
            raise ValueError("grid_shape must contain two positive integers") from error
        if not 0 < number <= np.iinfo(np.int32).max:
            raise ValueError("grid_shape values must fit positive int32")
        result.append(number)
    return tuple(result)


def _grid_coordinates(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    return np.column_stack((np.repeat(np.arange(rows), cols), np.tile(np.arange(cols), rows))).astype(np.int32)


def _grid_adjacency(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    count = rows * cols
    adjacency = np.zeros((count, count), dtype=np.uint8)
    for index in range(count):
        row, col = divmod(index, cols)
        if row + 1 < rows:
            adjacency[index, index + cols] = adjacency[index + cols, index] = 1
        if col + 1 < cols:
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1
    return adjacency


def _validate_adjacency(value: object, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.shape != (count, count):
        raise ValueError("adjacency must be a uint8 array of shape (n_buttons, n_buttons)")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("adjacency must be binary")
    if np.any(np.diag(array)) or not np.array_equal(array, array.T):
        raise ValueError("adjacency must be symmetric with a zero diagonal")
    return array


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True, eq=False)
class PublicStructure:
    """Immutable learner-facing structure; arrays use explicit v1 dtypes.

    ``structure_sha256`` includes the task name, ordered structure and axis
    conventions. ``rule_id`` identifies only the ordered logical rule.
    Display coordinates are deliberately absent from this format.
    """

    task_name: str
    family: str
    n_buttons: int
    rule_id: str
    grid_shape: np.ndarray | None = None
    grid_coordinates: np.ndarray | None = None
    adjacency: np.ndarray | None = None
    structure_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise ValueError("task_name must be a nonempty string")
        if self.family not in ("grid", "graph"):
            raise ValueError("family must be 'grid' or 'graph'")
        if isinstance(self.n_buttons, (bool, np.bool_)):
            raise ValueError("n_buttons must be a positive integer")
        try:
            count = operator.index(self.n_buttons)
        except TypeError as error:
            raise ValueError("n_buttons must be a positive integer") from error
        if count <= 0 or count * count > _MAX_ARCHIVE_BYTES:
            raise ValueError("n_buttons must be positive and fit the structure size limit")
        object.__setattr__(self, "n_buttons", count)
        if not isinstance(self.rule_id, str) or re.fullmatch(r"[0-9a-f]{64}", self.rule_id) is None:
            raise ValueError("rule_id must be a lowercase SHA256 hex string")
        if self.family == "grid":
            if self.adjacency is not None:
                raise ValueError("grid structures contain grid_shape/grid_coordinates, not adjacency")
            shape_array = np.asarray(self.grid_shape)
            if shape_array.dtype != np.dtype(np.int32) or shape_array.shape != (2,):
                raise ValueError("grid_shape must be an int32 array of shape (2,)")
            shape = _grid_shape(shape_array)
            if shape[0] * shape[1] != count:
                raise ValueError("grid_shape product must equal n_buttons")
            coordinates = np.asarray(self.grid_coordinates)
            if coordinates.dtype != np.dtype(np.int32) or coordinates.shape != (count, 2):
                raise ValueError("grid_coordinates must be an int32 array of shape (n_buttons, 2)")
            if not np.array_equal(coordinates, _grid_coordinates(shape)):
                raise ValueError("grid_coordinates must contain the complete row-major (row, column) grid")
            adjacency = _grid_adjacency(shape)
            object.__setattr__(self, "grid_shape", _immutable(shape_array))
            object.__setattr__(self, "grid_coordinates", _immutable(coordinates))
        else:
            if self.grid_shape is not None or self.grid_coordinates is not None:
                raise ValueError("graph structures contain adjacency, not grid fields")
            adjacency = _validate_adjacency(self.adjacency, count)
            object.__setattr__(self, "adjacency", _immutable(adjacency))
        if _logical_rule_id(adjacency) != self.rule_id:
            raise ValueError("rule_id does not match the ordered structure")
        digest = hashlib.sha256()
        digest.update(_canonical_json(self._metadata()))
        for name, array in sorted(self.observation_fields().items()):
            digest.update(name.encode("ascii") + b"\0")
            # Canonical little endian bytes make the content hash portable.
            digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
        object.__setattr__(self, "structure_sha256", digest.hexdigest())

    def observation_fields(self) -> dict[str, np.ndarray]:
        """Return independent copies of the public observation fields."""
        if self.family == "grid":
            return {"grid_shape": self.grid_shape.copy(), "grid_coordinates": self.grid_coordinates.copy()}
        return {"adjacency": self.adjacency.copy()}

    def _metadata(self) -> dict:
        return {
            "schema": STRUCTURE_SCHEMA,
            "task_name": self.task_name,
            "family": self.family,
            "n_buttons": self.n_buttons,
            "rule_id": self.rule_id,
            "semantics": _SEMANTICS,
            "arrays": {name: {"dtype": array.dtype.name, "shape": list(array.shape)}
                       for name, array in self.observation_fields().items()},
        }


def export_structure(rule: PuzzleRule) -> PublicStructure:
    """Copy a rule's declared public structure without inferring its name."""
    if rule.grid_shape is not None:
        shape = _grid_shape(rule.grid_shape)
        return PublicStructure(rule.name, "grid", rule.n_buttons, rule.rule_id,
                               grid_shape=np.array(shape, dtype=np.int32),
                               grid_coordinates=_grid_coordinates(shape))
    return PublicStructure(rule.name, "graph", rule.n_buttons, rule.rule_id,
                           adjacency=np.array(rule.adjacency, dtype=np.uint8, copy=True))


def _paths(path: str | os.PathLike) -> tuple[Path, Path]:
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("structure path must end in .npz")
    return destination, destination.with_suffix(".json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_structure(structure: PublicStructure, path: str | os.PathLike) -> Path:
    """Save an NPZ and matching JSON manifest without replacing either file.

    Both temporary files are complete before publishing with exclusive hard
    links. Publication of a two-file pair cannot be one atomic filesystem
    operation; a failed second publication removes only our first link.
    """
    if not isinstance(structure, PublicStructure):
        raise TypeError("structure must be a PublicStructure")
    destination, manifest_path = _paths(path)
    if os.path.lexists(destination) or os.path.lexists(manifest_path):
        raise FileExistsError("structure archive or manifest already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".structure-", suffix=".npz", delete=False) as stream:
            archive_temp = Path(stream.name)
            temporary_paths.append(archive_temp)
            np.savez_compressed(stream, **structure.observation_fields())
            stream.flush()
            os.fsync(stream.fileno())
        metadata = structure._metadata()
        metadata.update(structure_sha256=structure.structure_sha256,
                        npz_sha256=_file_sha256(archive_temp))
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".structure-", suffix=".json", delete=False) as stream:
            manifest_temp = Path(stream.name)
            temporary_paths.append(manifest_temp)
            stream.write(_canonical_json(metadata) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(archive_temp, destination)
        try:
            os.link(manifest_temp, manifest_path)
        except BaseException:
            # Avoid removing a replacement another process installed meanwhile.
            if destination.exists() and os.path.samefile(archive_temp, destination):
                destination.unlink()
            raise
        return destination
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_structure(path: str | os.PathLike) -> PublicStructure:
    """Read strict v1 pure data; never construct a rule, environment or solver."""
    archive_path, manifest_path = _paths(path)
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("structure manifest exceeds size limit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_json_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON value {value}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid structure manifest JSON") from error
    required = {"schema", "task_name", "family", "n_buttons", "rule_id", "semantics", "arrays", "structure_sha256", "npz_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("structure manifest has missing or unexpected fields")
    if manifest["schema"] != STRUCTURE_SCHEMA or manifest["semantics"] != _SEMANTICS:
        raise ValueError("unsupported structure schema or semantics")
    count = manifest["n_buttons"]
    if type(count) is not int or count <= 0 or count * count > _MAX_ARCHIVE_BYTES:
        raise ValueError("manifest n_buttons must be a positive integer within the size limit")
    if manifest["family"] not in ("grid", "graph"):
        raise ValueError("family must be 'grid' or 'graph'")
    for name in ("npz_sha256", "structure_sha256"):
        if not isinstance(manifest[name], str) or re.fullmatch(r"[0-9a-f]{64}", manifest[name]) is None:
            raise ValueError(f"invalid {name}")
    if archive_path.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise ValueError("structure archive exceeds size limit")
    if _file_sha256(archive_path) != manifest["npz_sha256"]:
        raise ValueError("structure archive file hash mismatch")
    expected = {"grid_shape", "grid_coordinates"} if manifest["family"] == "grid" else {"adjacency"}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != {name + ".npy" for name in expected}:
                raise ValueError("structure archive has missing, duplicate or unexpected arrays")
            if sum(entry.file_size for entry in entries) > _MAX_ARCHIVE_BYTES:
                raise ValueError("uncompressed structure archive exceeds size limit")
            expected_shapes = {"grid_shape": (2,), "grid_coordinates": (count, 2), "adjacency": (count, count)}
            for entry in entries:
                name = entry.filename[:-4]
                with archive.open(entry) as array_file:
                    version = np.lib.format.read_magic(array_file)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(array_file)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(array_file)
                    else:
                        raise ValueError("unsupported structure array format")
                    expected_dtype = np.dtype(np.uint8 if name == "adjacency" else np.int32)
                    if shape != expected_shapes[name] or dtype != expected_dtype:
                        raise ValueError("structure array header has invalid shape or dtype")
                    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    if entry.file_size != array_file.tell() + expected_bytes:
                        raise ValueError("structure array byte count does not match its header")
        with np.load(archive_path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in expected}
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("invalid structure archive") from error
    structure = PublicStructure(manifest["task_name"], manifest["family"], manifest["n_buttons"], manifest["rule_id"], **fields)
    if _canonical_json(manifest["arrays"]) != _canonical_json(structure._metadata()["arrays"]):
        raise ValueError("structure array metadata does not match data")
    if manifest["structure_sha256"] != structure.structure_sha256:
        raise ValueError("structure content hash mismatch")
    return structure


def rule_from_structure(structure: PublicStructure) -> PuzzleRule:
    """Explicitly build an executable rule for preparation or evaluation."""
    from .rules import PuzzleRule

    if not isinstance(structure, PublicStructure):
        raise TypeError("structure must be a PublicStructure")
    if structure.family == "grid":
        shape = tuple(map(int, structure.grid_shape))
        display_coordinates = structure.grid_coordinates[:, ::-1].astype(np.float64)
        return PuzzleRule(structure.task_name, _grid_adjacency(shape), display_coordinates, grid_shape=shape)
    return PuzzleRule(structure.task_name, structure.adjacency)
