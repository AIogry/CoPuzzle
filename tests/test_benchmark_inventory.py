"""Full-scope inventory qualification without any learned final-test evaluation."""
import json

import pytest

from copuzzle.benchmark.collection import freeze_data, public_descriptor, validate_inventory


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("v1-inventory") / "data"
    freeze_data(root)
    return root


def test_full_prespecified_inventory_and_semantics(frozen):
    report = validate_inventory(frozen, semantic=True)
    assert report["valid"] and report["bundles"] == 36
    assert report["training_episodes"] == 703076
    assert report["evaluation_cases"] == 220434
    descriptor = public_descriptor(frozen)
    assert descriptor["data_seeds"] == [101, 202, 303]
    assert len(descriptor["datasets"]) == 36
    assert "pair_codes" not in json.dumps(descriptor)
    assert "prefix_boards" not in json.dumps(descriptor)
    assert all(row["cell_populations"] for row in descriptor["datasets"])
    with pytest.raises(FileExistsError):
        freeze_data(frozen)


def test_missing_fit_partition_is_rejected_and_inventory_restored(frozen):
    path = frozen / "inventory.json"
    original = path.read_bytes()
    inventory = json.loads(original)
    inventory["bundles"] = inventory["bundles"][:-1]
    try:
        path.write_text(json.dumps(inventory))
        with pytest.raises(ValueError, match="combination"):
            validate_inventory(frozen)
    finally:
        path.write_bytes(original)


def test_inventory_detects_changed_case_bytes(frozen):
    path = next((frozen / "data").glob("*/learner/G_test.npz"))
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"not-a-valid-frozen-artifact")
        with pytest.raises(ValueError, match="checksum"):
            validate_inventory(frozen)
    finally:
        path.write_bytes(original)
