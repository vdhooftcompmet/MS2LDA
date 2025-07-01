import base64
import gzip
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest

from App import callbacks
from ms2lda.Mass2Motif import Mass2Motif


def test_calculate_motif_shares_mixed():
    spec = {
        "mz": [110.0],
        "intensities": [1.0],
        "metadata": {"id": "s1", "precursor_mz": 200.0},
    }
    lda = {
        "theta": {"s1": {"motif_1": 0.5, "motif_2": 0.5}},
        "beta": {
            "motif_1": {"frag@110": 0.6},
            "motif_2": {"loss@90": 0.4},
        },
    }
    shares = callbacks.calculate_motif_shares(spec, lda, tolerance=0.01)
    assert shares[0] == pytest.approx({"motif_1": 0.6, "motif_2": 0.4})  # noqa: S101


def test_make_spectrum_plot_none():
    spec = {"mz": [100.0], "intensities": [1.0], "metadata": {"id": "a"}}
    fig = callbacks.make_spectrum_plot(spec, None, {}, highlight_mode="none")
    assert isinstance(fig, go.Figure)  # noqa: S101
    assert fig.data[0].marker.color == "#7f7f7f"  # noqa: S101


def test_apply_common_layout():
    fig = go.Figure()
    callbacks.apply_common_layout(fig, ytitle="Intensity")
    assert fig.layout.bargap == 0.35  # noqa: S101,PLR2004
    assert fig.layout.yaxis.title.text == "Intensity"  # noqa: S101


def test_load_motifset_file(monkeypatch, tmp_path):
    def fake_load(_path):
        return pd.DataFrame(), pd.DataFrame()

    def fake_convert(_df):
        return ["m1"]

    monkeypatch.setattr(callbacks, "load_motifDB", fake_load)
    monkeypatch.setattr(callbacks, "motifDB2motifs", fake_convert)
    f = tmp_path / "file.json"
    f.write_text("{}")
    res = callbacks.load_motifset_file(str(f))
    assert res == ["m1"]  # noqa: S101


def test_toggle_tab_content():
    res = callbacks.toggle_tab_content("load-results-tab")
    assert res[1] == {"display": "block"}  # noqa: S101


def test_update_output():
    out = callbacks.update_output("data", "x.txt")
    assert "Uploaded File" in out.children[0].children  # noqa: S101


def test_toggle_advanced_settings():
    assert callbacks.toggle_advanced_settings(1, is_open=False)  # noqa: S101
    assert not callbacks.toggle_advanced_settings(None, is_open=False)  # noqa: S101


def test_parse_ms2lda_viz_file_roundtrip():
    data = {"a": 1}
    raw = json.dumps(data).encode()
    enc = base64.b64encode(raw).decode()
    content = f"data:application/json;base64,{enc}"
    assert callbacks.parse_ms2lda_viz_file(content) == data  # noqa: S101

    gz_bytes = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_bytes, mode="w") as gz:
        gz.write(raw)
    enc_gz = base64.b64encode(gz_bytes.getvalue()).decode()
    content_gz = f"data:application/json;base64,{enc_gz}"
    assert callbacks.parse_ms2lda_viz_file(content_gz) == data  # noqa: S101


def test_create_cytoscape_elements_simple():
    spectrum = Mass2Motif(
        frag_mz=np.array([100.0]),
        frag_intensities=np.array([0.8]),
        loss_mz=np.array([50.0]),
        loss_intensities=np.array([0.4]),
        metadata={
            "precursor_mz": 150.0,
            "losses": [{"loss_mz": 50.0, "loss_intensity": 0.4}],
        },
    )
    elems = callbacks.create_cytoscape_elements([spectrum], [], intensity_threshold=0.1)
    ids = {e["data"]["id"] for e in elems if "id" in e.get("data", {})}
    assert "motif_0" in ids  # noqa: S101
    assert "frag_100.0" in ids  # noqa: S101
    assert "loss_50.0" in ids  # noqa: S101


def test_compute_motif_degrees():
    lda = {
        "beta": {"motif_a": {}, "motif_b": {}},
        "theta": {
            "doc1": {"motif_a": 0.6, "motif_b": 0.2},
            "doc2": {"motif_a": 0.5, "motif_b": 0.1},
        },
        "overlap_scores": {
            "doc1": {"motif_a": 0.2, "motif_b": 0.1},
            "doc2": {"motif_a": 0.3, "motif_b": 0.05},
        },
    }
    res = callbacks.compute_motif_degrees(lda, 0.4, 1.0, 0.1, 0.3)
    assert res[0][0] == "motif_a"  # noqa: S101
    assert res[0][1] == 2  # noqa: S101,PLR2004
    assert res[1][1] == 0  # noqa: S101


def _simple_specs():
    return [
        {
            "mz": [150.0, 120.0],
            "intensities": [1.0, 0.5],
            "metadata": {"id": "s1", "precursor_mz": 300.0},
        },
        {
            "mz": [100.0],
            "intensities": [1.0],
            "metadata": {
                "id": "s2",
                "precursor_mz": 250.0,
                "losses": [{"loss_mz": 40.225}],
            },
        },
    ]


def test_update_spectra_search_table_frag_numeric():
    rows, _ = callbacks.update_spectra_search_table(
        _simple_specs(), "frag@150.00", [0, 1000],
    )
    assert len(rows) == 1  # noqa: S101
    assert rows[0]["spec_id"] == "s1"  # noqa: S101


def test_update_spectra_search_table_loss_numeric_tolerance():
    rows, _ = callbacks.update_spectra_search_table(
        _simple_specs(), "loss@40.234", [0, 1000],
    )
    assert len(rows) == 1  # noqa: S101
    assert rows[0]["spec_id"] == "s2"  # noqa: S101


def test_run_massql_query(monkeypatch):
    motifs = [
        {"mz": [100.0], "intensities": [1.0], "metadata": {"id": "motif_1"}},
        {"mz": [150.0], "intensities": [1.0], "metadata": {"id": "motif_2"}},
    ]

    def fake_process(_query, **_kwargs):
        return pd.DataFrame({"motif_id": ["motif_2"]})

    monkeypatch.setattr(callbacks.msql_engine, "process_query", fake_process)
    res = callbacks.run_massql_query(1, "QUERY", motifs)
    assert res == ["motif_2"]  # noqa: S101


def test_update_motif_rankings_table_massql_filter():
    lda = {
        "beta": {"motif_a": {}, "motif_b": {}},
        "theta": {"doc": {"motif_a": 0.6, "motif_b": 0.3}},
        "overlap_scores": {"doc": {"motif_a": 0.2, "motif_b": 0.1}},
    }
    rows, cols, msg = callbacks.update_motif_rankings_table(
        lda,
        [0, 1],
        [0, 1],
        "motif-rankings-tab",
        ["motif_a"],
        None,
        None,
    )
    assert len(rows) == 1  # noqa: S101
    assert rows[0]["Motif"] == "motif_a"  # noqa: S101

