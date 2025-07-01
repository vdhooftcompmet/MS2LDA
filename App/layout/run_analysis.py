import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

import os
from App.app_instance import SPEC2VEC_DIR

# Determine whether to show the Run Analysis tab. When the
# environment variable ``ENABLE_RUN_ANALYSIS`` is set to ``0`` or
# ``false`` the tab will be hidden. Default is to show it.
SHOW_RUN_ANALYSIS = os.getenv("ENABLE_RUN_ANALYSIS", "1").lower() not in (
    "0",
    "false",
)


def create_run_analysis_tab(show_tab: bool = True):
    """Create the Run Analysis tab.

    Parameters
    ----------
    show_tab: bool, optional
        If ``False`` the container is hidden using ``display: none``. The
        tab's contents are still created so callbacks referencing the
        components remain valid.
    """

    return html.Div(
        id="run-analysis-tab-content",
        children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab allows you to run an MS2LDA analysis from scratch using a single uploaded data file.
                        You can control basic parameters like the number of motifs, polarity, and top N Spec2Vec matches,
                        as well as advanced settings (e.g., min_mz, max_mz).
                        When ready, click "Run Analysis" to generate the results and proceed to the other tabs for visualization.
                        """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            # Data upload & basic MS2LDA parameters section
            html.Div(
                [
                    html.H4("Data Upload & Basic MS2LDA Parameters", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    ["Drag and Drop or ", html.A("Select Files")],
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=False,
                            ),
                            html.Div(
                                id="file-upload-info", style={"marginBottom": "20px", "textAlign": "center"},
                            ),
                            # Basic parameters (with tooltips):
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Number of Motifs", id="n-motifs-tooltip",
                                    ),
                                    dbc.Input(
                                        id="n-motifs",
                                        type="number",
                                        value=200,
                                        min=1,
                                    ),
                                ],
                                className="mb-3",
                                id="n-motifs-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Number of LDA topics to discover. Typically 10-100.",
                                target="n-motifs-tooltip",
                                placement="right",
                            ),
                            html.Div(
                                [
                                    dbc.Label(
                                        "Acquisition Type", id="acq-type-tooltip",
                                    ),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "DDA", "value": "DDA"},
                                            {"label": "DIA", "value": "DIA"},
                                        ],
                                        value="DDA",
                                        id="acquisition-type",
                                        inline=True,
                                    ),
                                ],
                                className="mb-3",
                                id="acq-type-div",
                            ),
                            dbc.Tooltip(
                                "Type of acquisition: DDA or DIA. Affects how the spectra are tokenized.",
                                target="acq-type-tooltip",
                                placement="right",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Top N Matches", id="topn-tooltip",
                                    ),
                                    dbc.Input(
                                        id="top-n", type="number", value=5, min=1,
                                    ),
                                ],
                                className="mb-3",
                                id="topn-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Number of library matches to retrieve per motif via Spec2Vec.",
                                target="topn-tooltip",
                                placement="right",
                            ),
                            html.Div(
                                [
                                    dbc.Label(
                                        "Unique Molecules", id="uniqmols-tooltip",
                                    ),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Yes", "value": True},
                                            {"label": "No", "value": False},
                                        ],
                                        value=True,
                                        id="unique-mols",
                                        inline=True,
                                    ),
                                ],
                                className="mb-3",
                                id="uniqmols-div",
                            ),
                            dbc.Tooltip(
                                "Whether to keep only unique compounds or include duplicates in Spec2Vec hits.",
                                target="uniqmols-tooltip",
                                placement="right",
                            ),
                            html.Div(
                                [
                                    dbc.Label("Polarity", id="polarity-tooltip"),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Positive", "value": "positive"},
                                            {"label": "Negative", "value": "negative"},
                                        ],
                                        value="positive",
                                        id="polarity",
                                        inline=True,
                                    ),
                                ],
                                className="mb-3",
                                id="polarity-div",
                            ),
                            dbc.Tooltip(
                                "Polarity can be used for specialized processing. Currently set to DDA.",
                                target="polarity-tooltip",
                                placement="right",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Iterations", id="iterations-tooltip",
                                    ),
                                    dbc.Input(
                                        id="n-iterations", type="number", value=10000,
                                    ),
                                ],
                                className="mb-3",
                                id="iterations-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Number of LDA training iterations. Higher = more thorough training.",
                                target="iterations-tooltip",
                                placement="right",
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
            # Spec2Vec Setup section
            html.Div(
                [
                    html.H4("Spec2Vec Setup", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                **Spec2Vec** is used to annotate motifs by comparing them against a pretrained model
                                and library embeddings. You can download the necessary files here if you haven't already.
                                If the files already exist, the process will skip them.
                                """,
                            ),
                            dcc.Loading(
                                id="download-s2v-spinner",
                                type="circle",
                                children=[
                                    dbc.Button(
                                        "📥 Download Spec2Vec Files",
                                        id="download-s2v-button",
                                        color="primary",
                                        className="mt-3",
                                    ),
                                    html.Div(
                                        id="download-s2v-status",
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                            ),
                            html.Br(),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "S2V Model Path", id="s2v-model-tooltip",
                                    ),
                                    dbc.Input(
                                        id="s2v-model-path",
                                        type="text",
                                        value=str(SPEC2VEC_DIR / "150225_Spec2Vec_pos_CleanedLibraries.model"),
                                    ),
                                ],
                                className="mb-3",
                                id="s2v-model-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Spec2Vec model file (trained embeddings). Provide full path.",
                                target="s2v-model-tooltip",
                                placement="right",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "S2V Library Embeddings",
                                        id="s2v-library-embeddings-tooltip",
                                    ),
                                    dbc.Input(
                                        id="s2v-library-embeddings",
                                        type="text",
                                        value=str(SPEC2VEC_DIR / "150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy"),
                                    ),
                                ],
                                className="mb-3",
                                id="s2v-library-embeddings-ig",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "S2V Library DB", id="s2v-library-db-tooltip",
                                    ),
                                    dbc.Input(
                                        id="s2v-library-db",
                                        type="text",
                                        value=str(SPEC2VEC_DIR / "150225_CombLibraries_spectra.db"),
                                    ),
                                ],
                                className="mb-3",
                                id="s2v-library-db-ig",
                            ),
                            dbc.Tooltip(
                                "Pickled library embeddings for Spec2Vec. Provide full path.",
                                target="s2v-library-tooltip",
                                placement="right",
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
            # Advanced Settings section
            html.Div(
                [
                    html.H4("Advanced Settings", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                These parameters allow fine-grained control over MS2LDA preprocessing,
                                convergence criteria, and model hyperparameters. Generally,
                                you can leave them as defaults unless you want to experiment
                                with more specialized behaviors or custom thresholding.
                                """,
                            ),
                            dbc.Button(
                                ["⚙️ Advanced Settings"],
                                id="advanced-settings-button",
                                color="secondary",
                                className="mb-3",
                            ),
                            dbc.Collapse(
                                id="advanced-settings-collapse",
                                is_open=False,
                                children=[
                                    html.H5("Advanced Parameters", className="mt-3", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6("Preprocessing", style={"fontSize": "16px", "fontWeight": "bold", "color": "#3c4c5e", "marginBottom": "6px"}),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "min_mz",
                                                                id="prep-minmz-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-min-mz",
                                                                type="number",
                                                                value=0,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-minmz-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Minimum m/z to keep in each spectrum’s peaks.",
                                                        target="prep-minmz-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "max_mz",
                                                                id="prep-maxmz-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-max-mz",
                                                                type="number",
                                                                value=2000,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-maxmz-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Maximum m/z to keep in each spectrum’s peaks.",
                                                        target="prep-maxmz-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "max_frags",
                                                                id="prep-maxfrags-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-max-frags",
                                                                type="number",
                                                                value=1000,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-maxfrags-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Max number of peaks (frags) to retain per spectrum.",
                                                        target="prep-maxfrags-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "min_frags",
                                                                id="prep-minfrags-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-min-frags",
                                                                type="number",
                                                                value=5,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-minfrags-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Minimum number of peaks needed to keep a spectrum.",
                                                        target="prep-minfrags-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "min_intensity",
                                                                id="prep-minint-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-min-intensity",
                                                                type="number",
                                                                value=0.01,
                                                                step=0.001,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-minint-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Lower intensity threshold (relative). Discard peaks below this fraction.",
                                                        target="prep-minint-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "max_intensity",
                                                                id="prep-maxint-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-max-intensity",
                                                                type="number",
                                                                value=1,
                                                                step=0.1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-maxint-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Upper intensity threshold (relative). Discard peaks above this fraction.",
                                                        target="prep-maxint-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6("Convergence", style={"fontSize": "16px", "fontWeight": "bold", "color": "#3c4c5e", "marginBottom": "6px"}),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "step_size",
                                                                id="conv-stepsz-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="conv-step-size",
                                                                type="number",
                                                                value=50,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="conv-stepsz-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Steps per iteration chunk before checking convergence criteria.",
                                                        target="conv-stepsz-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "window_size",
                                                                id="conv-winsz-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="conv-window-size",
                                                                type="number",
                                                                value=10,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="conv-winsz-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Number of iteration checkpoints to consider for convergence trend.",
                                                        target="conv-winsz-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "threshold",
                                                                id="conv-thresh-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="conv-threshold",
                                                                type="number",
                                                                value=0.005,
                                                                step=0.0001,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="conv-thresh-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Convergence threshold for perplexity or entropy changes.",
                                                        target="conv-thresh-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "type",
                                                                id="conv-type-tooltip",
                                                            ),
                                                            dbc.Select(
                                                                id="conv-type",
                                                                options=[
                                                                    {
                                                                        "label": "perplexity_history",
                                                                        "value": "perplexity_history",
                                                                    },
                                                                    {
                                                                        "label": "entropy_history_doc",
                                                                        "value": "entropy_history_doc",
                                                                    },
                                                                    {
                                                                        "label": "entropy_history_topic",
                                                                        "value": "entropy_history_topic",
                                                                    },
                                                                    {
                                                                        "label": "log_likelihood_history",
                                                                        "value": "log_likelihood_history",
                                                                    },
                                                                ],
                                                                value="perplexity_history",
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="conv-type-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Which metric to watch for early stopping: perplexity, doc/topic entropy, or log likelihood.",
                                                        target="conv-type-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                    ),
                                    html.Hr(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6("Annotation", style={"fontSize": "16px", "fontWeight": "bold", "color": "#3c4c5e", "marginBottom": "6px"}),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "criterium",
                                                                id="ann-criterium-tooltip",
                                                            ),
                                                            dbc.Select(
                                                                id="ann-criterium",
                                                                options=[
                                                                    {
                                                                        "label": "best",
                                                                        "value": "best",
                                                                    },
                                                                    {
                                                                        "label": "biggest",
                                                                        "value": "biggest",
                                                                    },
                                                                ],
                                                                value="biggest",
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="ann-criterium-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Use 'best' (hit cluster is best matched) or 'biggest' (largest cluster).",
                                                        target="ann-criterium-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "cosine_similarity",
                                                                id="ann-cossim-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="ann-cosine-sim",
                                                                type="number",
                                                                value=0.90,
                                                                step=0.01,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="ann-cossim-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Cosine similarity threshold for motif-spectra optimization step.",
                                                        target="ann-cossim-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6("Model", style={"fontSize": "16px", "fontWeight": "bold", "color": "#3c4c5e", "marginBottom": "6px"}),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "rm_top",
                                                                id="model-rmtop-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="model-rm-top",
                                                                type="number",
                                                                value=0,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="model-rmtop-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Number of top words to remove globally (tomotopy param). Often 0.",
                                                        target="model-rmtop-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "min_cf",
                                                                id="model-mincf-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="model-min-cf",
                                                                type="number",
                                                                value=0,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="model-mincf-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Minimum corpus frequency to keep a token (tomotopy param). Usually 0.",
                                                        target="model-mincf-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "min_df",
                                                                id="model-mindf-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="model-min-df",
                                                                type="number",
                                                                value=3,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="model-mindf-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Minimum document frequency to keep a token. For big corpora.",
                                                        target="model-mindf-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "alpha",
                                                                id="model-alpha-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="model-alpha",
                                                                type="number",
                                                                value=0.6,
                                                                step=0.1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="model-alpha-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Dirichlet prior for doc-topic distributions. Lower => more specialized topics.",
                                                        target="model-alpha-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "eta",
                                                                id="model-eta-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="model-eta",
                                                                type="number",
                                                                value=0.01,
                                                                step=0.001,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="model-eta-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Dirichlet prior for topic-word distributions. Lower => more peaky topics.",
                                                        target="model-eta-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "seed",
                                                                id="model-seed-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="model-seed",
                                                                type="number",
                                                                value=42,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="model-seed-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Random seed for reproducibility.",
                                                        target="model-seed-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6("Train"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "parallel",
                                                                id="train-parallel-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="train-parallel",
                                                                type="number",
                                                                value=3,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="train-parallel-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "How many parallel threads for tomotopy training. Usually 1-4.",
                                                        target="train-parallel-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "workers",
                                                                id="train-workers-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="train-workers",
                                                                type="number",
                                                                value=0,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="train-workers-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Additional worker threads for certain tomotopy tasks. Typically 0 or 1.",
                                                        target="train-workers-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                    ),
                                    html.Hr(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6("Dataset"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "sig_digits",
                                                                id="prep-sigdig-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="prep-sigdig",
                                                                type="number",
                                                                value=2,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="prep-sigdig-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Significant number of digits for fragments and losses.",
                                                        target="prep-sigdig-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "charge",
                                                                id="dataset-charge-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="dataset-charge",
                                                                type="number",
                                                                value=1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="dataset-charge-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Set the assumed precursor charge. E.g. 1 or 2.",
                                                        target="dataset-charge-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "Run Name",
                                                                id="dataset-name-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="dataset-name",
                                                                type="text",
                                                                value="ms2lda_dashboard_run",
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="dataset-name-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Name/identifier for this run (used in output filenames).",
                                                        target="dataset-name-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "Output Folder",
                                                                id="dataset-outdir-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="dataset-output-folder",
                                                                type="text",
                                                                value="ms2lda_results",
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="dataset-outdir-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Folder path where intermediate results & output files are saved.",
                                                        target="dataset-outdir-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6("Fingerprint"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "fp_type",
                                                                id="fp-type-tooltip",
                                                            ),
                                                            dbc.Select(
                                                                id="fp-type",
                                                                options=[
                                                                    {
                                                                        "label": "rdkit",
                                                                        "value": "rdkit",
                                                                    },
                                                                    {
                                                                        "label": "maccs",
                                                                        "value": "maccs",
                                                                    },
                                                                    {
                                                                        "label": "pubchem",
                                                                        "value": "pubchem",
                                                                    },
                                                                    {
                                                                        "label": "ecfp",
                                                                        "value": "ecfp",
                                                                    },
                                                                ],
                                                                value="maccs",
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="fp-type-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Which fingerprint type to use for structural similarity in motif annotation.",
                                                        target="fp-type-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "fp threshold",
                                                                id="fp-threshold-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="fp-threshold",
                                                                type="number",
                                                                value=0.8,
                                                                step=0.1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="fp-threshold-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Tanimoto threshold for motif fingerprint annotation.",
                                                        target="fp-threshold-tooltip",
                                                        placement="right",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(
                                                                "Motif Parameter",
                                                                id="motif-param-tooltip",
                                                            ),
                                                            dbc.Input(
                                                                id="motif-parameter",
                                                                type="number",
                                                                value=50,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                        id="motif-param-ig",
                                                    ),
                                                    dbc.Tooltip(
                                                        "Number of top words to define each motif (e.g. 50).",
                                                        target="motif-param-tooltip",
                                                        placement="right",
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                    ),
                                ],
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
            # Run Analysis section
            html.Div(
                [
                    html.H4("Run Analysis", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                Once everything is configured, click **Run Analysis**
                                to perform LDA on your spectra. Depending on the dataset
                                size and iterations, this can take a while. Please wait until the
                                progress indicator has finished to retrieve the results.
                                """,
                            ),
                            dcc.Loading(
                                id="run-analysis-spinner",
                                type="circle",
                                children=[
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "▶️ Run Analysis",
                                                id="run-button",
                                                color="primary",
                                            ),
                                        ],
                                        className="d-grid gap-2 mt-3",
                                    ),
                                    html.Div(
                                        id="run-status", style={"marginTop": "20px"},
                                    ),
                                ],
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
        style={"display": "block" if show_tab else "none"},
    )

