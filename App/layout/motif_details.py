import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html



def create_motif_details_tab():
    return html.Div(
        id="motif-details-tab-content",
        children=dcc.Loading(
            id="motif-details-loading",
            type="circle",
            fullscreen=True,
            children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab provides detailed insight into a selected MS2LDA motif, highlighting possible
                        chemical structures, motif composition, and the actual spectra that represent it.
                        The content is structured into three clear sections: Motif Details, Features in Motifs,
                        and Spectra in Motifs. Each section contains explanations to help interpret the results.
                        """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            # ----------------------------------------------------------------
            # 1. MOTIF DETAILS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4(id="motif-details-title", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            html.H5("Spec2Vec Matching Results", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                            dcc.Markdown(
                                """
                        The Spec2Vec matching results displayed here suggest chemical structures (SMILES strings) that closely match the selected motif. Spec2Vec calculates similarities by comparing motif pseudo-spectra against reference spectra from a known database. Matches shown here can help identify possible chemical identities or provide clues about structural characteristics represented by this motif.
                        """,
                            ),
                            html.Div(
                                id="motif-spec2vec-container",
                                style={"marginTop": "10px"},
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
                            html.H5("Optimised vs Raw Motif Pseudo-Spectra", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                            dcc.Markdown(
                                """
                        This view compares two aligned versions of the selected Mass2Motif, highlighting changes made during optimisation.

                        The **top panel** displays the optimised pseudo-spectrum (relative intensity scale). It is constructed by aggregating fragments or losses consistently matched across high-quality library spectra identified by Spec2Vec. Being library-derived, this optimised spectrum is independent of the LDA probability thresholds and typically provides a cleaner representation.

                        The **bottom panel** shows the raw LDA pseudo-spectrum (probability scale), filtered according to your chosen thresholds. Higher bars indicate peaks strongly associated with this motif according to the LDA topic model.

                        Both panels share the same m/z axis, making it easy to spot retained or removed peaks during optimisation. Use the toggle below to switch between fragment and loss views.
                        """,
                            ),
                            dbc.RadioItems(
                                id="optimised-motif-fragloss-toggle",
                                options=[
                                    {"label": "Fragments + Losses", "value": "both"},
                                    {"label": "Fragments Only", "value": "fragments"},
                                    {"label": "Losses Only", "value": "losses"},
                                ],
                                value="both",
                                inline=True,
                            ),
                            dbc.Label("Bar / Line Thickness"),
                            dcc.Slider(
                                id="dual-plot-bar-width-slider",
                                min=0.1,
                                max=2.0,
                                step=0.1,
                                value=0.8,
                                marks={
                                    0.1: "0.1",
                                    0.5: "0.5",
                                    1: "1",
                                    1.5: "1.5",
                                    2.0: "2",
                                },
                                tooltip={"always_visible": False, "placement": "top"},
                            ),
                            html.Div(
                                id="motif-dual-spectrum-container",
                                style={"marginTop": "10px"},
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
            # 2. FEATURES IN MOTIFS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Features in Motifs", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            html.H5("Spectra-Peaks Probability Filter", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                            dcc.Markdown(
                                """
                        The slider below controls the minimum and maximum probability thresholds for selecting motif features (fragments and losses) identified by the LDA model. Setting a higher minimum value keeps only peaks strongly associated with the motif, providing a simpler representation. A lower minimum includes more peaks, potentially capturing additional detail but also introducing noise.
                        """,
                            ),
                            dbc.Label("Spectra-Peaks Probability Filter:"),
                            dcc.RangeSlider(
                                id="probability-filter",
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
                                allowCross=False,
                            ),
                            html.Div(
                                id="probability-filter-display",
                                style={"marginTop": "10px"},
                            ),
                            dcc.Markdown(
                                "Document-level filters (apply to table **and** bar-plot):",
                                style={"marginTop": "15px"},
                            ),
                            dbc.Label("Motif Probability (θ) Filter:"),
                            dcc.RangeSlider(
                                id="doc-topic-filter",
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
                                allowCross=False,
                            ),
                            html.Div(
                                id="doc-topic-filter-display",
                                style={"marginTop": "10px"},
                            ),
                            dbc.Label("Overlap Score Filter:"),
                            dcc.RangeSlider(
                                id="overlap-filter",
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
                                allowCross=False,
                            ),
                            html.Div(
                                id="overlap-filter-display", style={"marginTop": "10px"},
                            ),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div(
                        [
                            html.H5("Motif Features Table and Summary Plots", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                            dcc.Markdown(
                                """
                        The table below lists the motif features (fragments and losses) that pass the probability filter, including their probabilities within the motif. Below it, a bar plot shows how frequently each feature appears **within the filtered set of documents** for this motif (i.e., documents whose doc-topic probability and overlap score both pass the current threshold ranges).
                        """,
                            ),
                            html.Div(id="motif-features-container"),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "10px",
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
            # 3. SPECTRA IN MOTIFS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Spectra in Motifs", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                        This section lists MS2 spectra linked to the selected motif. When
                        you pick a spectrum (or use the navigation buttons) the plot underneath
                        updates.

                        Peaks matching fragment or neutral-loss features of the motif turn
                        red. For every neutral-loss match, a green dashed line connects the
                        fragment peak to the precursor ion and is annotated with the
                        loss value. All remaining peaks are grey. The plot lets you judge whether the
                        motif's characteristic features are genuinely present in the experimental spectrum.
                        """,
                            ),
                            html.Div(id="motif-documents-container"),
                            dash_table.DataTable(
                                id="spectra-table",
                                data=[],
                                columns=[],
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left"},
                                page_size=10,
                                filter_action="native",
                                row_selectable="single",
                                selected_rows=[0],
                                hidden_columns=["SpecIndex"],
                                style_data_conditional=[
                                    {
                                        'if': {'state': 'active'},
                                        'backgroundColor': 'transparent',
                                        'border': 'transparent'
                                    }
                                ],
                            ),
                            html.Div(
                                [
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "🔍 All motifs",
                                                id="spectrum-highlight-all-btn",
                                                color="primary",
                                                outline=True,
                                                active=False,
                                                className="me-1",
                                            ),
                                            dbc.Button(
                                                "❌ None",
                                                id="spectrum-highlight-none-btn",
                                                color="primary",
                                                outline=True,
                                                active=False,
                                            ),
                                        ],
                                        className="me-2",
                                    ),
                                    dbc.RadioItems(
                                        id="spectrum-fragloss-toggle",
                                        options=[
                                            {
                                                "label": "Fragments + Losses",
                                                "value": "both",
                                            },
                                            {
                                                "label": "Fragments Only",
                                                "value": "fragments",
                                            },
                                            {"label": "Losses Only", "value": "losses"},
                                        ],
                                        value="both",
                                        inline=True,
                                        style={"marginLeft": "10px"},
                                    ),
                                    dbc.Checkbox(
                                        id="spectrum-show-parent-ion",
                                        label="Show Parent Ion",
                                        value=True,
                                        className="ms-3",
                                    ),
                                ],
                                className="d-flex align-items-center flex-wrap mb-2",
                            ),
                            html.Div(
                                [
                                    html.H5("Individual motifs (probability):", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                                    html.Div(
                                        id="motif-details-associated-motifs-list",
                                        style={"marginTop": "5px"},
                                    ),
                                ],
                            ),
                            # Hidden input to store the highlight mode
                            dcc.Store(id="spectrum-highlight-mode", data="single"),
                            html.Div(id="spectrum-plot"),
                            html.Div(
                                [
                                    dbc.Button(
                                        "⬅️ Previous",
                                        id="prev-spectrum",
                                        n_clicks=0,
                                        color="primary",
                                    ),
                                    dbc.Button(
                                        "➡️ Next",
                                        id="next-spectrum",
                                        n_clicks=0,
                                        className="ms-2",
                                        color="primary",
                                    ),
                                    dbc.Button(
                                        "Spectrum Details ↗",
                                        id="jump-to-search-btn",
                                        color="primary",
                                        className="ms-2",
                                    ),
                                ],
                                className="mt-3",
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
        ],
        ),
        style={"display": "none"},
    )


