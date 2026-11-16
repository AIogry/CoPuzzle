import importlib.util
import json
from pathlib import Path

import pytest

from copuzzle_baselines.study.report import build_report, load_analysis

spec = importlib.util.spec_from_file_location("report_rendering_fixture", Path(__file__).resolve().parents[1] / "examples/qualify_report.py")
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
synthetic_analysis = fixture_module.synthetic_analysis


def test_report_keeps_qualification_labels_and_binds_input_files(tmp_path):
    source = synthetic_analysis(tmp_path / "analysis")
    result = build_report(source, tmp_path / "report")
    assert result["synthetic"] and result["role"] == "qualification"
    text = (tmp_path / "report/REPORT.md").read_text()
    assert "Synthetic rendering fixture" in text
    assert "Lights Out 3×3" in text and "Native rule" in text
    assert "gcbc/c100/b0" in text and "no native R/C" in text
    with pytest.raises(FileExistsError):
        build_report(source, tmp_path / "report")
    (source / "rows.json").write_text('[]')
    with pytest.raises(ValueError, match="checksum"):
        load_analysis(source)


def test_report_rejects_formal_label_on_synthetic_data(tmp_path):
    source = synthetic_analysis(tmp_path / "analysis")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["role"], manifest["split"] = "formal", "test"
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="synthetic"):
        load_analysis(source)


def test_report_requires_hashes_for_every_input_table(tmp_path):
    source = synthetic_analysis(tmp_path / "analysis")
    manifest = json.loads((source / "manifest.json").read_text())
    del manifest["files"]["statistics.json"]
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hashes"):
        load_analysis(source)


def test_scientific_figures_are_saved_with_their_rendering_source(tmp_path):
    pytest.importorskip("matplotlib")
    source = synthetic_analysis(tmp_path / "analysis")
    result = build_report(source, tmp_path / "report", figures=True)
    assert result["render_versions"]["matplotlib"]
    assert len(list((tmp_path / "report/figures").glob("*.png"))) == 3
    assert len(list((tmp_path / "report/figures").glob("*.svg"))) == 3
    assert (tmp_path / "report/report_source.py").is_file()
