# App/callbacks/__init__.py
"""Callbacks module initialization.

This module imports all callback functions to ensure they are registered
with the Dash app when the callbacks module is imported.
"""

# Import all callback modules to register their callbacks
from App.callbacks import common
from App.callbacks import run_and_load
from App.callbacks import network
from App.callbacks import rankings_details
from App.callbacks import screening

# Make common utilities available at package level
from App.callbacks.common import (
    apply_common_layout,
    make_spectrum_plot,
    calculate_motif_shares,
    load_motifset_file,
    MOTIFDB_FOLDER,
)

__all__ = [
    "common",
    "run_and_load", 
    "network",
    "rankings_details",
    "screening",
    "apply_common_layout",
    "make_spectrum_plot",
    "calculate_motif_shares",
    "load_motifset_file",
    "MOTIFDB_FOLDER",
]