import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html



def create_motif_rankings_tab():
    return html.Div(
        id="motif-rankings-tab-content",
        children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab displays all discovered motifs in a ranked table format. You can filter motifs based on 
                        probability and overlap thresholds, or search for specific motifs using MassQL queries.
                        The table shows each motif's degree (number of documents containing it), average probability, 
                        and average overlap score. Click on any motif to view its detailed composition and associated spectra.
                        """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            dbc.Container(
                [
                    # ----------------------------------------------------------------
                    # 1. FILTER CONTROLS
                    # ----------------------------------------------------------------
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Filter Controls", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px", "display": "inline-block"}),
                                    dbc.Button(
                                        "🔍 Hide",
                                        id="filter-controls-toggle-button",
                                        color="primary",
                                        size="sm",
                                        className="ms-2",
                                        style={"display": "inline-block", "marginLeft": "10px", "marginBottom": "5px"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "width": "100%"},
                            ),
                            dbc.Collapse(
                                id="filter-controls-collapse",
                                is_open=True,
                                children=[
                                    html.Div(
                                        [
                                            # Show/Hide Explanation button
                                            dbc.Button(
                                                "ℹ️ Info",
                                                id="motif-rankings-explanation-button",
                                                color="secondary",
                                                size="sm",
                                                className="mb-3",
                                            ),
                                            # Collapsible explanation section
                                            dbc.Collapse(
                                                id="motif-rankings-explanation-collapse",
                                                is_open=False,
                                                children=[
                                                    dcc.Markdown(
                                                        """
                                                        From here you can filter your motifs in a ranked table based on how many documents (spectra) meet the selected Probability and Overlap ranges. For each motif, we compute a `Degree` representing the number of documents where the motif's doc-topic probability and overlap score both fall within the selected threshold ranges. We also report an `Average Doc-Topic Probability` and an `Average Overlap Score`. These averages are computed only over the documents that pass the filters, so they can be quite high if the motif strongly dominates the docs where it appears. The `Overlap Score` is computed by multiplying the word-topic distribution with the topic-word distribution for each word in a document, and then summing these products for each topic. This provides a measure of how well a document's word distribution matches a topic's expected word distribution. Adjust the two RangeSliders below to narrow the doc-level thresholds on Probability and Overlap. _A motif remains in the table only if at least one document passes these filters_. Clicking on a motif row takes you to a detailed view of that motif.

                                                        You can also use [MassQL (Mass Spectrometry Query Language)](https://www.nature.com/articles/s41592-025-02660-z) to filter and search for specific motifs. MassQL uses SQL-like syntax to query mass spectrometry data, enabling discovery of chemical patterns across datasets. In this application, motifs are translated into pseudo-spectra where fragments and losses are treated as peaks, allowing you to search for specific fragment masses or filter motifs by metadata. The filtering process applies all filters in sequence: first, motifs are filtered based on the Probability and Overlap thresholds, and then the MassQL query is applied to the remaining motifs. For example, you can use queries like `QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_123` to filter spectra by a specific motif ID, or `QUERY scaninfo(MS2DATA) WHERE MS2PROD=178.03` to find spectra containing a specific product ion mass.
                                                        """,
                                                        style={
                                                            "backgroundColor": "#f8f9fa",
                                                            "padding": "15px",
                                                            "borderRadius": "5px",
                                                            "border": "1px solid #e9ecef",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Average Doc-Topic Probability",
                                                                style={"fontWeight": "bold"},
                                                            ),
                                                            dcc.RangeSlider(
                                                                id="probability-thresh",
                                                                min=0,
                                                                max=1,
                                                                step=0.01,
                                                                value=[0, 1],
                                                                marks={
                                                                    0: "0",
                                                                    0.25: "0.25",
                                                                    0.5: "0.5",
                                                                    0.75: "0.75",
                                                                    1: "1",
                                                                },
                                                                tooltip={
                                                                    "always_visible": False,
                                                                    "placement": "top",
                                                                },
                                                                allowCross=False,
                                                            ),
                                                            html.Div(
                                                                id="probability-thresh-display",
                                                                style={"marginTop": "10px"},
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Average Overlap Score",
                                                                style={"fontWeight": "bold"},
                                                            ),
                                                            dcc.RangeSlider(
                                                                id="overlap-thresh",
                                                                min=0,
                                                                max=1,
                                                                step=0.01,
                                                                value=[0, 1],
                                                                marks={
                                                                    0: "0",
                                                                    0.25: "0.25",
                                                                    0.5: "0.5",
                                                                    0.75: "0.75",
                                                                    1: "1",
                                                                },
                                                                tooltip={
                                                                    "always_visible": False,
                                                                    "placement": "top",
                                                                },
                                                                allowCross=False,
                                                            ),
                                                            html.Div(
                                                                id="overlap-thresh-display",
                                                                style={"marginTop": "10px"},
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                            ),
                                            dbc.Label(
                                                "MassQL Query",
                                                style={"fontWeight": "bold", "marginTop": "20px"},
                                            ),
                                            dbc.Textarea(
                                                id="motif-ranking-massql-input",
                                                placeholder="Enter your MassQL query here",
                                                style={"width": "100%", "height": "150px", "marginTop": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    "🔍 Run Query",
                                                                    id="motif-ranking-massql-btn",
                                                                    color="primary",
                                                                ),
                                                                width="auto",
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    "🔄 Reset Query",
                                                                    id="motif-ranking-massql-reset-btn",
                                                                    color="primary",
                                                                ),
                                                                width="auto",
                                                            ),
                                                        ],
                                                        className="mt-3",
                                                    ),
                                                ],
                                            ),
                                            dcc.Store(id="motif-ranking-massql-matches"),
                                            html.Div(
                                                id="motif-ranking-massql-error",
                                                style={
                                                    "marginTop": "10px",
                                                    "color": "#dc3545",
                                                    "padding": "10px",
                                                    "backgroundColor": "#f8d7da",
                                                    "borderRadius": "5px",
                                                    "display": "none"
                                                },
                                            ),
                                            html.Div(
                                                id="motif-rankings-count",
                                                style={
                                                    "marginTop": "20px", 
                                                    "fontWeight": "bold",
                                                    "fontSize": "16px",
                                                    "color": "#007bff",
                                                    "padding": "10px",
                                                    "backgroundColor": "#f8f9fa",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center"
                                                },
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
                    # 2. MOTIF RANKINGS TABLE
                    # ----------------------------------------------------------------
                    html.Div(
                        [
                            html.H4("Motif Rankings Table", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                            html.Div(
                                [
                                    dash_table.DataTable(
                                        id="motif-rankings-table",
                                        data=[],
                                        columns=[],
                                        sort_action="custom",
                                        filter_action="native",
                                        page_size=20,
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "minWidth": "150px",
                                            "width": "200px",
                                            "maxWidth": "400px",
                                            "whiteSpace": "normal",
                                            "textAlign": "left",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"column_id": "Motif"},
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
                                        style_header={
                                            "backgroundColor": "rgb(230, 230, 230)",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    dbc.Button(
                                        "📄 Save to CSV",
                                        id="save-motifranking-csv",
                                        color="primary",
                                        className="mt-2",
                                    ),
                                    dbc.Button(
                                        "💾 Save to JSON",
                                        id="save-motifranking-json",
                                        color="primary",
                                        className="ms-2 mt-2",
                                    ),
                                    dcc.Download(id="download-motifranking-csv"),
                                    dcc.Download(id="download-motifranking-json"),
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
            ),
        ],
        style={"display": "none"},
    )


