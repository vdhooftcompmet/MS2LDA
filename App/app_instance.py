import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash_extensions.enrich import DashProxy, ServersideOutputTransform, FileSystemBackend
from pathlib import Path
import ms2lda

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "ms2lda_cache"
CACHE_DIR.mkdir(exist_ok=True)

# shared cache directory (adjust path if /tmp is cleared on reboot)
cache = FileSystemBackend(cache_dir=str(CACHE_DIR))

app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    transforms=[ServersideOutputTransform(backends=[cache])],
)
app.title = "MS2LDA Interactive Dashboard"

PKG_ROOT = Path(ms2lda.__file__).resolve().parent
SPEC2VEC_DIR = PKG_ROOT / "Add_On" / "Spec2Vec" / "model_positive_mode"
FPCALC_DIR = PKG_ROOT / "Add_On" / "Fingerprints" / "FP_calculation"
MOTIFDB_DIR = PKG_ROOT / "MotifDB"

# Include Cytoscape extra layouts
cyto.load_extra_layouts()
