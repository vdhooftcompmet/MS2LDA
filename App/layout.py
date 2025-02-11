import dash_bootstrap_components as dbc
from dash import dash_table
from dash import html, dcc


def create_run_analysis_tab():
    tab = html.Div(
        id="run-analysis-tab-content",
        children=[
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab allows you to run an MS2LDA analysis from scratch using a single uploaded data file. 
                        You can control basic parameters like the number of motifs, polarity, and top N Spec2Vec matches, 
                        as well as advanced settings (e.g., min_mz, max_mz). 
                        When ready, click "Run Analysis" to generate the results and proceed to the other tabs for visualization.
                        """
                    )
                ],
                style={"margin-top": "20px", "margin-bottom": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    ["Drag and Drop or ", html.A("Select Files")]
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
                                id="file-upload-info", style={"marginBottom": "20px"}
                            ),
                            # Basic parameters (with tooltips):
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Number of Motifs", id="n-motifs-tooltip"),
                                    dbc.Input(
                                        id="n-motifs",
                                        type="number",
                                        value=50,
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
                                    dbc.Label("Acquisition Type", id="acq-type-tooltip"),
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
                                    dbc.InputGroupText("Top N Matches", id="topn-tooltip"),
                                    dbc.Input(
                                        id="top-n", type="number", value=5, min=1
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
                                    dbc.Label("Unique Molecules", id="uniqmols-tooltip"),
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
                                    dbc.InputGroupText("Iterations", id="iterations-tooltip"),
                                    dbc.Input(id="n-iterations", type="number", value=1000),
                                ],
                                className="mb-3",
                                id="iterations-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Number of LDA training iterations. Higher = more thorough training.",
                                target="iterations-tooltip",
                                placement="right",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("S2V Model Path", id="s2v-model-tooltip"),
                                    dbc.Input(
                                        id="s2v-model-path",
                                        type="text",
                                        value="../MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model",
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
                                    dbc.InputGroupText("S2V Library Path", id="s2v-library-tooltip"),
                                    dbc.Input(
                                        id="s2v-library-path",
                                        type="text",
                                        value="../MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl",
                                    ),
                                ],
                                className="mb-3",
                                id="s2v-library-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Pickled library embeddings for Spec2Vec. Provide full path.",
                                target="s2v-library-tooltip",
                                placement="right",
                            ),
                            dbc.Button(
                                "Show/Hide Advanced Settings",
                                id="advanced-settings-button",
                                color="info",
                                className="mb-3",
                            ),
                            dbc.Collapse(
                                id="advanced-settings-collapse",
                                is_open=False,
                                children=[
                                    html.H5("Advanced Parameters", className="mt-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    # Preprocessing
                                                    html.H6("Preprocessing"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("min_mz", id="prep-minmz-tooltip"),
                                                            dbc.Input(
                                                                id="prep-min-mz",
                                                                type="number",
                                                                value=0,  # default
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
                                                            dbc.InputGroupText("max_mz", id="prep-maxmz-tooltip"),
                                                            dbc.Input(
                                                                id="prep-max-mz",
                                                                type="number",
                                                                value=2000,  # default
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
                                                            dbc.InputGroupText("max_frags",
                                                                               id="prep-maxfrags-tooltip"),
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
                                                            dbc.InputGroupText("min_frags",
                                                                               id="prep-minfrags-tooltip"),
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
                                                            dbc.InputGroupText("min_intensity",
                                                                               id="prep-minint-tooltip"),
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
                                                            dbc.InputGroupText("max_intensity",
                                                                               id="prep-maxint-tooltip"),
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
                                                    # Convergence
                                                    html.H6("Convergence"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("step_size",
                                                                               id="conv-stepsz-tooltip"),
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
                                                            dbc.InputGroupText("window_size",
                                                                               id="conv-winsz-tooltip"),
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
                                                            dbc.InputGroupText("threshold",
                                                                               id="conv-thresh-tooltip"),
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
                                                            dbc.InputGroupText("type", id="conv-type-tooltip"),
                                                            dbc.Select(
                                                                id="conv-type",
                                                                options=[
                                                                    {"label": "perplexity_history",
                                                                     "value": "perplexity_history"},
                                                                    {"label": "entropy_history_doc",
                                                                     "value": "entropy_history_doc"},
                                                                    {"label": "entropy_history_topic",
                                                                     "value": "entropy_history_topic"},
                                                                    {"label": "log_likelihood_history",
                                                                     "value": "log_likelihood_history"},
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
                                                    html.H6("Annotation"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("criterium",
                                                                               id="ann-criterium-tooltip"),
                                                            dbc.Select(
                                                                id="ann-criterium",
                                                                options=[
                                                                    {"label": "best", "value": "best"},
                                                                    {"label": "biggest", "value": "biggest"},
                                                                ],
                                                                value="best",
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
                                                            dbc.InputGroupText("cosine_similarity",
                                                                               id="ann-cossim-tooltip"),
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
                                                    html.H6("Model"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("rm_top", id="model-rmtop-tooltip"),
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
                                                            dbc.InputGroupText("min_cf", id="model-mincf-tooltip"),
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
                                                            dbc.InputGroupText("min_df", id="model-mindf-tooltip"),
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
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("alpha", id="model-alpha-tooltip"),
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
                                                            dbc.InputGroupText("eta", id="model-eta-tooltip"),
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
                                                            dbc.InputGroupText("seed", id="model-seed-tooltip"),
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
                                                            dbc.InputGroupText("parallel",
                                                                               id="train-parallel-tooltip"),
                                                            dbc.Input(
                                                                id="train-parallel",
                                                                type="number",
                                                                value=1,
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
                                                            dbc.InputGroupText("workers",
                                                                               id="train-workers-tooltip"),
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
                                                    # n_iterations Moved Above
                                                ],
                                                width=6,
                                            ),
                                        ]
                                    ),
                                    html.Hr(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6("Dataset"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("sig_digits",
                                                                               id="prep-sigdig-tooltip"),
                                                            dbc.Input(
                                                                id="prep-sigdig",
                                                                type="number",
                                                                value=2,  # default
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
                                                            dbc.InputGroupText("charge",
                                                                               id="dataset-charge-tooltip"),
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
                                                            dbc.InputGroupText("Run Name",
                                                                               id="dataset-name-tooltip"),
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
                                                            dbc.InputGroupText("Output Folder",
                                                                               id="dataset-outdir-tooltip"),
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
                                                            dbc.InputGroupText("fp_type", id="fp-type-tooltip"),
                                                            dbc.Select(
                                                                id="fp-type",
                                                                options=[
                                                                    {"label": "rdkit", "value": "rdkit"},
                                                                    {"label": "maccs", "value": "maccs"},
                                                                    {"label": "pubchem", "value": "pubchem"},
                                                                    {"label": "ecfp", "value": "ecfp"},
                                                                ],
                                                                value="rdkit",
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
                                                            dbc.InputGroupText("fp threshold",
                                                                               id="fp-threshold-tooltip"),
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
                                                            dbc.InputGroupText("Motif Parameter",
                                                                               id="motif-param-tooltip"),
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
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Run Analysis",
                                        id="run-button",
                                        color="primary",
                                    ),
                                ],
                                className="d-grid gap-2",
                            ),
                            html.Div(id="run-status", style={"marginTop": "20px"}),
                        ],
                        width=6,
                    )
                ],
                justify="center",
            )
        ],
        style={"display": "block"},
    )
    return tab


def create_load_results_tab():
    tab = html.Div(
        id="load-results-tab-content",
        children=[
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab allows you to load previously generated MS2LDA results (a JSON file). 
                        Once loaded, you can explore them immediately in the subsequent tabs. 
                        This is useful if you’ve run an analysis before and want to revisit or share the results.
                        """
                    )
                ],
                style={"margin-top": "20px", "margin-bottom": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Upload(
                                id="upload-results",
                                children=html.Div(
                                    ["Drag and Drop or ", html.A("Select Results File")]
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
                            html.Div(id="load-status", style={"marginTop": "20px"}),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Load Results",
                                        id="load-results-button",
                                        color="primary",
                                    ),
                                ],
                                className="d-grid gap-2 mt-3",
                            ),
                        ],
                        width=6,
                    )
                ],
                justify="center",
            )
        ],
        style={"display": "none"},
    )
    return tab


def create_cytoscape_network_tab():
    tab = html.Div(
        id="results-tab-content",
        children=[
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab shows an interactive network of optimized motifs. 
                        Each motif is displayed as a node, and its fragments or losses 
                        appear as connected nodes (color-coded for clarity). 
                        Only edges above the selected intensity threshold will be shown. 
                        You can adjust that threshold with the slider. 
                        By default, the extra edges from each loss node to its 
                        corresponding fragment node are hidden for less clutter, 
                        but you can re-enable them using the checkbox. 
                        Click on any motif node to see its associated molecules on the right side.
                        """
                    )
                ],
                style={"margin-top": "20px", "margin-bottom": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Edge Intensity Threshold"),
                        dcc.Slider(
                            id="edge-intensity-threshold",
                            min=0,
                            max=1,
                            step=0.05,
                            value=0.50,
                            marks={0: "0.0", 0.5: "0.5", 1: "1.0"}
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Checklist(
                            options=[{"label": "Add Loss -> Fragment Edge", "value": "show_loss_edge"}],
                            value=[],
                            id="toggle-loss-edge",
                            inline=True,
                        )
                    ], width=6),
                ],
                style={"marginTop": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Graph Layout"),
                        dcc.Dropdown(
                            id="cytoscape-layout-dropdown",
                            options=[
                                {"label": "CoSE", "value": "cose"},
                                {"label": "Force-Directed (Spring)", "value": "fcose"},
                                {"label": "Circle", "value": "circle"},
                                {"label": "Concentric", "value": "concentric"},
                            ],
                            value="fcose",
                            clearable=False,
                        )
                    ], width=6),
                ],
                style={"marginTop": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                id="cytoscape-network-container",
                                style={
                                    "marginTop": "20px",
                                    "height": "600px",
                                },
                            )
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                id="molecule-images",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "20px",
                                    "overflowY": "auto",
                                    "height": "600px",
                                    "padding": "10px",
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "5px",
                                },
                            ),
                        ],
                        width=4,
                    ),
                ],
                align="start",
                className="g-3",
            )
        ],
        style={"display": "none"},
    )
    return tab


def create_motif_rankings_tab():
    tab = html.Div(
        id="motif-rankings-tab-content",
        children=[
            dbc.Container([
                html.Div(
                    [
                        dcc.Markdown(
                            """
                            This tab displays your motifs in a ranked table, based on how frequently they appear 
                            in your data and their average probabilities. You can filter the table by probability or overlap thresholds 
                            to highlight motifs of interest. Clicking on a motif in the table takes you to detailed information about that motif.
                            """
                        )
                    ],
                    style={"margin-top": "20px", "margin-bottom": "20px"},
                ),

                dbc.Row([
                    dbc.Col([
                        # Row for the sliders and their displays
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Probability Threshold Range"),
                                        dcc.RangeSlider(
                                            id="probability-thresh",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=[0.1, 1],
                                            marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                            tooltip={"always_visible": False, "placement": "top"},
                                            allowCross=False
                                        ),
                                        html.Div(id='probability-thresh-display', style={"marginTop": "10px"})
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Overlap Threshold Range"),
                                        dcc.RangeSlider(
                                            id="overlap-thresh",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=[0.3, 1],
                                            marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                            tooltip={"always_visible": False, "placement": "top"},
                                            allowCross=False
                                        ),
                                        html.Div(id='overlap-thresh-display', style={"marginTop": "10px"})
                                    ],
                                    width=6
                                )
                            ]
                        ),
                        html.Div(id="motif-rankings-table-container", style={"marginTop": "20px"})
                    ], width=12),
                ]),
            ]),
        ],
        style={"display": "none"}
    )
    return tab


def create_motif_details_tab():
    tab = html.Div(
        id="motif-details-tab-content",
        children=[
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab is organized into three main sections: (1) Motif Details, (2) Features in Motifs, and (3) Documents in Motifs. 

                        In **Motif Details**, you'll see the Spec2Vec matching results for the selected motif.
                        In **Features in Motifs**, a probability filter helps you select which motif features (fragments/losses) to show, 
                        along with bar plots indicating how strongly these features belong to the motif and how often they appear in the dataset.
                        Finally, **Documents in Motifs** displays the spectra containing this motif, with filters for document-topic probability and overlap score.
                        """
                    )
                ],
                style={"margin-top": "20px", "margin-bottom": "20px"},
            ),
            html.Div(
                [
                    html.H4(id='motif-details-title'),
                    dcc.Markdown(
                        """
                        This section shows SMILES structures found by comparing the motif’s pseudo-spectrum against an external library.
                        You can view top matching compounds here and visually inspect their chemical structures.
                        """
                    ),
                    # This container will be dynamically filled in the callback
                    html.Div(id='motif-spec2vec-container'),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "margin-bottom": "5px"
                },
            ),
            html.Div(
                [
                    html.H4("Features in Motifs"),
                    dcc.Markdown(
                        """
                        The **Topic-Word Probability Filter** below controls which motif features (fragments/losses) are displayed 
                        based on their probability (`beta`). After filtering, you’ll see a table of selected features plus 
                        bar charts illustrating their distribution in the motif and the dataset.
                        """
                    ),

                    dbc.Label("Topic-Word Probability Filter:"),
                    dcc.RangeSlider(
                        id='probability-filter',
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0, 1],
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                        allowCross=False
                    ),
                    html.Div(id='probability-filter-display', style={"marginTop": "10px"}),

                    # This container will hold the feature table & first bar chart
                    html.Div(id='motif-features-container'),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "margin-bottom": "5px"
                },
            ),
            html.Div(
                [
                    html.H4("Documents in Motifs"),
                    dcc.Markdown(
                        """
                        This section shows spectra that include the current motif. 
                        **Document-Topic Probability Filter** (theta) narrows the list by motif representation in each spectrum, 
                        and **Overlap Score Filter** focuses on how closely the spectrum’s features match this motif’s top features.
                        """
                    ),

                    dbc.Label("Document-Topic Probability Filter:"),
                    dcc.RangeSlider(
                        id='doc-topic-filter',
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0, 1],
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                        allowCross=False
                    ),
                    html.Div(id='doc-topic-filter-display', style={"marginTop": "10px"}),

                    dbc.Label("Overlap Score Filter:"),
                    dcc.RangeSlider(
                        id='overlap-filter',
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0, 1],
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                        allowCross=False
                    ),
                    html.Div(id='overlap-filter-display', style={"marginTop": "10px"}),

                    # This container will hold the second bar chart & doc table
                    html.Div(id='motif-documents-container'),

                    dash_table.DataTable(
                        id='spectra-table',
                        data=[],
                        columns=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        page_size=10,
                        row_selectable='single',
                        selected_rows=[0],
                        hidden_columns=["SpecIndex"],
                    ),

                    html.Div(id='spectrum-plot'),

                    html.Div([
                        dbc.Button('Previous', id='prev-spectrum', n_clicks=0, color="info"),
                        dbc.Button('Next', id='next-spectrum', n_clicks=0, className='ms-2', color="info"),
                    ], className='mt-3'),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "margin-bottom": "5px"
                },
            ),
        ],
        style={"display": "none"},
    )
    return tab


def create_screening_tab():
    return html.Div(
        id="screening-tab-content",
        style={"display": "none"},
        children=[
            html.Div(
                [
                    dcc.Markdown("""
                        This tab allows you to automatically compare your optimized motifs
                        against the reference motifs from MotifDB. To begin, first select 
                        which reference sets you want to include. Then click "Compute Similarities" 
                        to run the screening using Spec2Vec comparison. Screening results are shown 
                        in the table below, and you can use the slider to filter the table by minimum similarity score.
                    """),
                ],
                style={"margin-top": "20px", "margin-bottom": "20px"},
            ),
            html.Hr(),
            html.H4("Reference Motif Sets Found"),
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
                    )
                ],
            ),
            dbc.Button("Compute Similarities", id="compute-screening-button", color="primary", disabled=False),
            dbc.Progress(id="screening-progress", value=0, striped=True, animated=True,
                         style={"marginTop": "10px", "width": "100%", "height": "20px"}),
            html.Div(id="compute-screening-status", style={"marginTop": "10px"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Label("Minimum Similarity Score"),
                    dcc.Slider(
                        id="screening-threshold-slider",
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.0,
                        marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                    ),
                    html.Div(id="screening-threshold-value", style={"marginTop": "10px"}),
                ], width=6),
            ], style={"marginTop": "10px"}),
            html.H5("Screening Results (Filtered)"),
            dash_table.DataTable(
                id="screening-results-table",
                columns=[
                    {"name": "User Motif ID", "id": "user_motif_id"},
                    {"name": "User ShortAnno", "id": "user_short_annotation"},
                    {"name": "Reference Motif ID", "id": "ref_motif_id"},
                    {"name": "Ref ShortAnno", "id": "ref_short_annotation"},
                    {"name": "Ref MotifSet", "id": "ref_motifset"},
                    {"name": "Similarity Score", "id": "score"},
                ],
                data=[],
                page_size=15,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "maxWidth": "250px", "whiteSpace": "normal"},
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                # NEW: make user_motif_id visually clickable
                style_data_conditional=[
                    {
                        'if': {'column_id': 'user_motif_id'},
                        'cursor': 'pointer',
                        'textDecoration': 'underline',
                        'color': 'blue',
                    },
                ],
            ),
            dbc.Button("Save to CSV", id="save-screening-csv", color="secondary", className="mt-2"),
            dbc.Button("Save to JSON", id="save-screening-json", color="secondary", className="ms-2 mt-2"),
            dcc.Download(id="download-screening-csv"),
            dcc.Download(id="download-screening-json"),
        ],
    )
