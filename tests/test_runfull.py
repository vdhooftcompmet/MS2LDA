import sys
import types

import pytest

sys.modules.setdefault("faiss", types.SimpleNamespace())
pytest.importorskip("spec2vec", reason="requires spec2vec")
from scripts import ms2lda_runfull as runfull  # noqa: E402


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def update(self, n):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyResp:
    def __init__(self, content=b"data"):
        self.content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):  # noqa: ARG002
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def test_github_file_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runfull, "tqdm", DummyTqdm)
    monkeypatch.setattr(runfull.requests, "get", lambda *_a, **_k: DummyResp())
    out = tmp_path / "x.txt"
    res = runfull._github_file_download("repo", "file", out)  # noqa: SLF001
    assert out.read_bytes() == b"data"  # noqa: S101
    assert res.endswith("(downloaded)")  # noqa: S101


def test_github_dir_download(tmp_path, monkeypatch):
    calls = []

    def fake_file(_repo, path, dst):
        calls.append(path)
        dst.write_text("ok")
        return "done"

    monkeypatch.setattr(runfull, "_github_file_download", fake_file)

    class DirResp(DummyResp):
        def json(self):
            return [
                {"type": "file", "name": "a"},
                {"type": "file", "name": "b"},
            ]

    monkeypatch.setattr(runfull.requests, "get", lambda *_a, **_k: DirResp())
    out = tmp_path
    log = runfull._github_dir_download("repo", "dir", out)  # noqa: SLF001
    assert calls == ["dir/a", "dir/b"]  # noqa: S101
    assert log == ["done", "done"]  # noqa: S101


def test_download_all_aux_data(monkeypatch, tmp_path):
    monkeypatch.setattr(runfull, "_github_file_download", lambda _r, _p, d: str(d))
    monkeypatch.setattr(runfull, "_github_dir_download", lambda _r, _d, _l: ["x"])
    monkeypatch.setattr(
        runfull,
        "download_model_and_data",
        lambda mode="positive": "spec2vec",  # noqa: ARG005
    )
    dummy_pkg = types.SimpleNamespace(__file__=str(tmp_path / "x"))
    monkeypatch.setattr(runfull, "ms2lda", dummy_pkg)
    res = runfull.download_all_aux_data()
    assert "Spec2Vec assets" in res  # noqa: S101


def test_deep_update():
    src = {"a": {"b": 1}, "c": 2}
    upd = {"a": {"b": 3, "d": 4}, "e": 5}
    res = runfull.deep_update(src, upd)
    assert res["a"]["b"] == 3  # noqa: S101,PLR2004
    assert res["a"]["d"] == 4  # noqa: S101,PLR2004
    assert res["c"] == 2  # noqa: S101,PLR2004
    assert res["e"] == 5  # noqa: S101,PLR2004


def _call_main(monkeypatch, _tmp_path, args):
    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(runfull, "run_ms2lda", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    monkeypatch.setattr(runfull, "download_all_aux_data", lambda: None)
    runfull.main()
    return called


def test_main_sets_basename(tmp_path, monkeypatch):
    dataset = tmp_path / "test.mgf"
    dataset.write_text("")
    output = tmp_path / "out"
    args = [
        "--dataset",
        str(dataset),
        "--n-motifs",
        "2",
        "--n-iterations",
        "3",
        "--output-folder",
        str(output),
    ]
    called = _call_main(monkeypatch, tmp_path, args)
    assert called["dataset_parameters"]["name"] == "ms2lda_test"  # noqa: S101
    assert called["dataset_parameters"]["output_folder"] == str(output)  # noqa: S101


def test_main_with_run_name(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.mgf"
    dataset.write_text("")
    output = tmp_path / "out"
    args = [
        "--dataset",
        str(dataset),
        "--n-motifs",
        "2",
        "--n-iterations",
        "3",
        "--output-folder",
        str(output),
        "--run-name",
        "foo",
    ]
    called = _call_main(monkeypatch, tmp_path, args)
    assert called["dataset_parameters"]["name"] == "foo"  # noqa: S101
    assert called["dataset_parameters"]["output_folder"] == str(output)  # noqa: S101
