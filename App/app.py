import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import App.callbacks  # noqa: F401 -- callbacks must be imported to register callbacks with the app
from App.layout import (
    create_run_analysis_tab,
    create_load_results_tab,
    create_cytoscape_network_tab,
    create_motif_rankings_tab,
    create_motif_details_tab,
    create_screening_tab,
    create_nts_tab,
    create_spectra_search_tab,
)
from App.app_instance import app  # Import the Dash app instance

ENABLE_RUN_ANALYSIS = os.getenv("ENABLE_RUN_ANALYSIS", "1").lower() not in (
    "0",
    "false",
)

server = app.server # gunicorn will import this WSGI callable

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src="assets/MS2LDA_LOGO_white.jpg",
                                    alt="MS2LDA Logo",
                                    height="100px",
                                    style={
                                        "display": "block",
                                        "margin": "0 auto",
                                    },
                                ),
                                html.A(
                                    "Docs 📚",
                                    href="https://vdhooftcompmet.github.io/MS2LDA/",
                                    target="_blank",
                                    style={
                                        "position": "absolute",
                                        "right": "20px",
                                        "top": "50%",
                                        "transform": "translateY(-50%)",
                                        "fontWeight": "bold",
                                        "fontSize": "18px",
                                        "textDecoration": "none",
                                    },
                                ),
                            ],
                            style={
                                "position": "relative",
                                "padding": "0 10px",
                            },
                        ),
                    ],
                    width=True,
                ),
            ],
            align="center",
        ),
        html.Hr(),
        dcc.Tabs(
            id="tabs",
            value="run-analysis-tab" if ENABLE_RUN_ANALYSIS else "load-results-tab",
            children=[
                *([
                    dcc.Tab(
                        label="Run Analysis",
                        value="run-analysis-tab",
                        id="run-analysis-tab",
                    )
                ] if ENABLE_RUN_ANALYSIS else []),
                dcc.Tab(
                    label="Load Results",
                    value="load-results-tab",
                    id="load-results-tab",
                ),
                dcc.Tab(
                    label="Motif Rankings",
                    value="motif-rankings-tab",
                    id="motif-rankings-tab",
                ),
                dcc.Tab(
                    label="Motif Details",
                    value="motif-details-tab",
                    id="motif-details-tab",
                ),
                dcc.Tab(
                    label="Spectra Search",
                    value="search-spectra-tab",
                    id="search-spectra-tab",
                ),
                dcc.Tab(label="View Network", 
                        value="results-tab", 
                        id="results-tab"
                ),
                dcc.Tab(label="Motif Search", 
                        value="screening-tab", 
                        id="screening-tab"
                ),
                dcc.Tab(label="nontarget screening",
                        value="nts-tab",
                        id="nts-tab"
                )
            ],
            className="mt-3",
        ),
        # Tabs for all the sections
        create_run_analysis_tab(show_tab=ENABLE_RUN_ANALYSIS),
        create_load_results_tab(),
        create_motif_rankings_tab(),
        create_motif_details_tab(),
        create_spectra_search_tab(),
        create_cytoscape_network_tab(),
        create_screening_tab(),
        create_nts_tab(),
        # Hidden storage
        dcc.Store(id="motif-spectra-ids-store"),
        dcc.Store(id="selected-spectrum-index", data=0),
        dcc.Store(id="clustered-smiles-store"),
        dcc.Store(id="optimized-motifs-store"),
        dcc.Store(id="lda-dict-store"),
        dcc.Store(id="selected-motif-store"),
        dcc.Store(id="spectra-store"),
        dcc.Store(id="mass2motifs-store"),
        dcc.Store(id="s2v-model-store"),
        dcc.Store(id="screening-fullresults-store"),
        dcc.Store(id="m2m-subfolders-store"),
        dcc.Store(id="motif-rankings-state", data=None, storage_type="memory"),
        dcc.Store(id="s2v-download-complete", data=False),
        # hidden output used only to trigger the client-side scroll
        html.Div(id="search-scroll-dummy", style={"display": "none"}),
        # hidden output used only to trigger client-side scroll
        html.Div(id="motif-details-scroll-dummy", style={"display": "none"}),
    ],
    fluid=False,
)

if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except dash.exceptions.ObsoleteAttributeException:
        app.run(debug=True)
