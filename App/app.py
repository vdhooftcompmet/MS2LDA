import dash_bootstrap_components as dbc
from dash import html, dcc

import layout as layout # without App.layout
from app_instance import app # Import the Dash app instance

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.Img(src="assets/MS2LDA_LOGO_white.jpg", alt="MS2LDA Logo", height="250px",
                         style={'display': 'block', 'margin': 'auto'}),
                dcc.Markdown("""
                Developed by [Jonas Dietrich](https://github.com/j-a-dietrich),
                [Rosina Torres Ortega](https://github.com/rtlortega), and 
                [Joe Wandy](https://github.com/joewandy).
                """, style={'textAlign': 'center'})
            ], width=True),
        ], align="end"),
        html.Hr(),

        dcc.Tabs(
            id="tabs",
            value="run-analysis-tab",
            children=[
                dcc.Tab(label="Run Analysis", value="run-analysis-tab", id="run-analysis-tab"),
                dcc.Tab(label="Load Results", value="load-results-tab", id="load-results-tab"),
                dcc.Tab(label="View Network", value="results-tab", id="results-tab"),
                dcc.Tab(label="Motif Rankings", value="motif-rankings-tab", id="motif-rankings-tab"),
                dcc.Tab(label="Motif Details", value="motif-details-tab", id="motif-details-tab"),
                dcc.Tab(label="Screening", value="screening-tab", id="screening-tab"),
            ],
            className="mt-3",
        ),

        # Tabs for all the sections
        layout.create_run_analysis_tab(),
        layout.create_load_results_tab(),
        layout.create_cytoscape_network_tab(),
        layout.create_motif_rankings_tab(),
        layout.create_motif_details_tab(),
        layout.create_screening_tab(),

        # Hidden storage
        dcc.Store(id='motif-spectra-ids-store'),
        dcc.Store(id='selected-spectrum-index', data=0),
        dcc.Store(id="clustered-smiles-store"),
        dcc.Store(id="optimized-motifs-store"),
        dcc.Store(id="lda-dict-store"),
        dcc.Store(id='selected-motif-store'),
        dcc.Store(id='spectra-store'),
        dcc.Store(id="screening-fullresults-store"),
        dcc.Store(id="m2m-subfolders-store"),
        dcc.Store(id="motif-rankings-state", data=None, storage_type="memory"),

    ],
    fluid=False,
)

if __name__ == "__main__":
    app.run_server(debug=True)
