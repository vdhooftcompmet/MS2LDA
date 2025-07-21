import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html



def create_screening_tab():
    return html.Div(
        id="screening-tab-content",
        style={"display": "none"},
        children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        r"""
                        This tab allows you to perform a **motif\-motif search**. Your discovered
                        motifs are compared against reference motifs from MotifDB to find potential 
                        matches and thereby annotate their chemical identity.
                        First select which reference sets you want to include, then
                        click "Compute Similarities" to run the motif search using
                        Spec2Vec comparison. Results are shown in the table below and can
                        be filtered by minimum similarity score using the slider.
                    """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            # ----------------------------------------------------------------
            # 1. REFERENCE MOTIF SELECTION
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Reference Motif Selection", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                Select which reference motif sets you want to include in your search.
                                These are the libraries of known motifs that your discovered motifs will be compared against.
                                """
                            ),
                            dcc.Loading(
                                id="m2m-subfolders-loading",
                                type="default",
                                children=[
                                    dbc.Checklist(
                                        id="m2m-folders-checklist",
                                        options=[],
                                        value=[],
                                        switch=True,
                                        className="mb-3",
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "🔄 Compute Similarities",
                                        id="compute-screening-button",
                                        color="primary",
                                        disabled=False,
                                    ),
                                ],
                                className="d-grid gap-2 mt-3",
                            ),
                            dbc.Progress(
                                id="screening-progress",
                                value=0,
                                striped=True,
                                animated=True,
                                style={"marginTop": "10px", "width": "100%", "height": "20px"},
                            ),
                            html.Div(id="compute-screening-status", style={"marginTop": "10px"}),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "15px",
                        },
                    ),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),
            # ----------------------------------------------------------------
            # 2. FILTERING CONTROLS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Filtering Controls", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                Use the slider below to filter the results by minimum similarity score.
                                Only matches with a similarity score above the threshold will be shown in the results table.
                                """
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Minimum Similarity Score", style={"fontWeight": "bold"}),
                                            dcc.Slider(
                                                id="screening-threshold-slider",
                                                min=0,
                                                max=1,
                                                step=0.05,
                                                value=0.0,
                                                marks={
                                                    0: "0",
                                                    0.25: "0.25",
                                                    0.5: "0.5",
                                                    0.75: "0.75",
                                                    1: "1",
                                                },
                                            ),
                                            html.Div(
                                                id="screening-threshold-value",
                                                style={"marginTop": "10px"},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "15px",
                        },
                    ),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),
            # ----------------------------------------------------------------
            # 3. SEARCH RESULTS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Motif Search Results", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                This table shows the matches between your discovered motifs and reference motifs.
                                Click on any user motif ID to view its details. You can sort and filter the table by any column.
                                """
                            ),
                            dash_table.DataTable(
                                id="screening-results-table",
                                columns=[
                                    {"name": "User Motif ID", "id": "user_motif_id"},
                                    {"name": "User AutoAnno", "id": "user_auto_annotation"},
                                    {"name": "Reference Motif ID", "id": "ref_motif_id"},
                                    {"name": "Ref ShortAnno", "id": "ref_short_annotation"},
                                    {"name": "Ref MotifSet", "id": "ref_motifset"},
                                    {"name": "Similarity Score", "id": "score", "type": "numeric", "format": {"specifier": ".4f"}},
                                ],
                                data=[],
                                page_size=20,
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "maxWidth": "250px",
                                    "whiteSpace": "normal",
                                },
                                style_header={
                                    "backgroundColor": "rgb(230, 230, 230)",
                                    "fontWeight": "bold",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "user_motif_id"},
                                        "cursor": "pointer",
                                        "textDecoration": "underline",
                                        "color": "blue",
                                    },
                                    {
                                        'if': {'state': 'active'},
                                        'backgroundColor': 'transparent',
                                        'border': 'transparent'
                                    }
                                ],
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "📄 Save to CSV",
                                        id="save-screening-csv",
                                        color="primary",
                                        className="mt-2",
                                    ),
                                    dbc.Button(
                                        "💾 Save to JSON",
                                        id="save-screening-json",
                                        color="primary",
                                        className="ms-2 mt-2",
                                    ),
                                ],
                                style={"marginTop": "10px"},
                            ),
                            dcc.Download(id="download-screening-csv"),
                            dcc.Download(id="download-screening-json"),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "15px",
                        },
                    ),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),
        ],
    )


