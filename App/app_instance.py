import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from pathlib import Path
import MS2LDA

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "MS2LDA Interactive Dashboard"

PKG_ROOT = Path(MS2LDA.__file__).resolve().parent
SPEC2VEC_DIR = PKG_ROOT / "Add_On" / "Spec2Vec" / "model_positive_mode"
FPCALC_DIR = PKG_ROOT / "Add_On" / "Fingerprints" / "FP_calculation"
MOTIFDB_DIR = PKG_ROOT / "MotifDB"

# Include Cytoscape extra layouts
cyto.load_extra_layouts()
