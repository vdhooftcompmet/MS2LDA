import tempfile
from pathlib import Path

from ms2lda import motif_parser
from ms2lda.utils import create_spectrum


def _motif():
    return create_spectrum([("frag@100", 1.0), ("loss@10", 0.5)], k=0)


def test_store_and_load_m2m_file_roundtrip():
    motif = _motif()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        motif_parser.store_m2m_file(motif, 0, str(tmp_path))
        filepath = next(tmp_path.iterdir())
        loaded = motif_parser.load_m2m_file(filepath)
        assert loaded.peaks.mz.tolist() == motif.peaks.mz.tolist()  # noqa: S101


def test_store_and_load_m2m_folder():
    motif = _motif()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        folder = tmp_path / "out"
        motif_parser.store_m2m_folder([motif], str(folder))
        loaded = motif_parser.load_m2m_folder(str(folder))
        assert len(loaded) == 1  # noqa: S101

