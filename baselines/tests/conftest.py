import pytest

from copuzzle.research.bundle import write_bundle


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle", "sliding-2x3", "state_holdout", seed=0, per_cell=1)
