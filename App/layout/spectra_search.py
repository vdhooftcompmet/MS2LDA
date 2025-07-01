import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html



def create_spectra_search_tab():
    return html.Div(
        id="search-spectra-tab-content",
        style={"display": "none"},
        children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab allows you to search for specific spectra in your dataset based on fragment/loss values or parent mass range.
                        You can filter spectra by entering a numeric value and selecting whether to search in fragments, losses, or both using the checkboxes,
                        or by specifying a parent mass range.
                        Click on any spectrum in the results table to view its detailed plot and associated motifs.
                        """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            # ----------------------------------------------------------------
            # 1. SEARCH CONTROLS
            # ----------------------------------------------------------------
            dbc.Container(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Search Controls", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px", "display": "inline-block"}),
                                    dbc.Button(
                                        "🔎 Hide",
                                        id="search-controls-toggle-button",
                                        color="primary",
                                        size="sm",
                                        className="ms-2",
                                        style={"display": "inline-block", "marginLeft": "10px", "marginBottom": "5px"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "width": "100%"},
                            ),
                            dbc.Collapse(
                                id="search-controls-collapse",
                                is_open=True,
                                children=[
                                    html.Div(
                                        [
                                            # Show/Hide Explanation button
                                            dbc.Button(
                                                "ℹ️ Info",
                                                id="spectra-search-explanation-button",
                                                color="secondary",
                                                size="sm",
                                                className="mb-3",
                                            ),
                                            # Collapsible explanation section
                                            dbc.Collapse(
                                                id="spectra-search-explanation-collapse",
                                                is_open=False,
                                                children=[
                                                    dcc.Markdown(
                                                        """
                                                        Search for specific spectra in your dataset based on fragment/loss values or parent mass range.

                                                        **Search by Fragment or Loss**: Enter a numeric value (e.g., 150.1) to search for spectra containing that specific fragment or loss mass. The search uses a tolerance of 0.01 Da. Use the checkboxes to specify whether to search in fragments, losses, or both.

                                                        **Parent Mass Range**: Use the slider to filter spectra based on their parent mass. This is useful for narrowing down your search to a specific mass range.

                                                        The results table shows matching spectra with their ID, parent mass, and lists of fragments and losses. Click on any spectrum to view its detailed plot and associated motifs below.
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
                                                            dbc.Label("Search by Fragment or Loss:", style={"fontWeight": "bold"}),
                                                            dbc.Input(
                                                                id="spectra-search-fragloss-input",
                                                                type="text",
                                                                placeholder="Enter a numeric value (e.g. 150.1)",
                                                            ),
                                                            dbc.Tooltip(
                                                                "Enter a numeric value for comparison with a tolerance of 0.01. "
                                                                "Use the checkboxes below to search in fragments, losses, or both.",
                                                                target="spectra-search-fragloss-input",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dbc.Checkbox(
                                                                        id="spectra-search-fragment-checkbox",
                                                                        label="Fragment",
                                                                        value=True,
                                                                        className="mr-2",
                                                                        style={"marginTop": "10px", "marginRight": "10px", "display": "inline-block"}
                                                                    ),
                                                                    dbc.Checkbox(
                                                                        id="spectra-search-loss-checkbox",
                                                                        label="Loss",
                                                                        value=True,
                                                                        className="mr-2",
                                                                        style={"marginTop": "10px", "display": "inline-block"}
                                                                    ),
                                                                ],
                                                                style={"marginTop": "5px"}
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Parent Mass Range:", style={"fontWeight": "bold"}),
                                                            dcc.RangeSlider(
                                                                id="spectra-search-parentmass-slider",
                                                                step=1,
                                                                allowCross=False,
                                                            ),
                                                            html.Div(
                                                                id="spectra-search-parentmass-slider-display",
                                                                style={"marginTop": "10px"},
                                                            ),
                                                        ],
                                                        width=8,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="spectra-search-status-message",
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
                ],
            ),
            # ----------------------------------------------------------------
            # 2. SEARCH RESULTS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Search Results", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dash_table.DataTable(
                                id="spectra-search-results-table",
                                columns=[
                                    {"name": "Spectrum ID", "id": "spec_id"},
                                    {"name": "Parent Mass", "id": "parent_mass"},
                                    {"name": "Feature ID", "id": "feature_id"},
                                    {"name": "Fragments", "id": "fragments"},
                                    {"name": "Losses", "id": "losses"},
                                ],
                                sort_action="custom",
                                sort_mode="single",
                                page_size=20,
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left", "whiteSpace": "normal"},
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "spec_id"},
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
            # 3. SPECTRUM DETAILS
            # ----------------------------------------------------------------
            dcc.Store(id="search-tab-selected-spectrum-details-store"),
            dcc.Store(id="search-tab-selected-motif-id-for-plot-store"),
            dcc.Store(id="search-highlight-mode", data="all"),
            html.Div(
                id="search-tab-spectrum-details-container",
                style={"marginTop": "20px", "display": "none"},
                children=dcc.Loading(
                    id="search-spectrum-details-loading",
                    type="circle",
                    fullscreen=True,
                    children=[
                        html.Div(
                            [
                                html.H4(id="search-tab-spectrum-title", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                                html.Div(
                                    [
                                        html.H5("Spectrum Visualization Controls", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                                        dcc.Markdown(
                                            """
                                            Control how the spectrum is displayed using the options below. You can highlight specific motifs,
                                            show only fragments or losses, and toggle the display of the parent ion.
                                            """
                                        ),
                                        html.Div(
                                            [
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button(
                                                            "🔍 All motifs",
                                                            id="search-highlight-all-btn",
                                                            color="primary",
                                                            outline=True,
                                                            active=True,
                                                            className="me-1",
                                                        ),
                                                        dbc.Button(
                                                            "❌ None",
                                                            id="search-highlight-none-btn",
                                                            color="primary",
                                                            outline=True,
                                                            active=False,
                                                        ),
                                                    ],
                                                    className="me-2",
                                                ),
                                                dbc.RadioItems(
                                                    id="search-fragloss-toggle",
                                                    options=[
                                                        {"label": "Fragments + Losses", "value": "both"},
                                                        {"label": "Fragments Only", "value": "fragments"},
                                                        {"label": "Losses Only", "value": "losses"},
                                                    ],
                                                    value="both",
                                                    inline=True,
                                                    style={"marginLeft": "10px"},
                                                ),
                                                dbc.Checkbox(
                                                    id="search-show-parent-ion",
                                                    label="Show Parent Ion",
                                                    value=True,
                                                    className="ms-3",
                                                ),
                                            ],
                                            className="d-flex align-items-center flex-wrap mb-2",
                                        ),
                                    ],
                                    style={
                                        "border": "1px dashed #ccc",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H5("Associated Motifs", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                                        dcc.Markdown(
                                            """
                                            This section shows the motifs associated with the selected spectrum and their probabilities.
                                            Click on a motif to highlight it in the spectrum plot below.
                                            """
                                        ),
                                        html.Div(
                                            id="search-tab-associated-motifs-list",
                                            style={"marginTop": "5px"},
                                        ),
                                    ],
                                    style={
                                        "border": "1px dashed #ccc",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H5("Spectrum Plot", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                                        html.Div(id="search-tab-spectrum-plot-container"),
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
            ),
        ],
    )
