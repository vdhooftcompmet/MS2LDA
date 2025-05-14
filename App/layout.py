import dash_bootstrap_components as dbc
from dash import dash_table
from dash import html, dcc


def create_run_analysis_tab():
    tab = html.Div(
        id="run-analysis-tab-content",
        children=[
            # Card 1: Intro / Explanation
            dbc.Card(
                [
                    dbc.CardHeader("MS2LDA Run Analysis Overview"),
                    dbc.CardBody(
                        [
                            dcc.Markdown(
                                """
                                This tab allows you to run an MS2LDA analysis from scratch using a single uploaded data file. 
                                You can control basic parameters like the number of motifs, polarity, and top N Spec2Vec matches, 
                                as well as advanced settings (e.g., min_mz, max_mz). 
                                When ready, click "Run Analysis" to generate the results and proceed to the other tabs for visualization.
                                """
                            ),
                        ]
                    ),
                ],
                style={"border": "1px solid #ccc", "padding": "10px", "marginTop": "20px", "marginBottom": "20px"},
            ),

            # Card 2: Data upload & basic MS2LDA parameters
            dbc.Card(
                [
                    dbc.CardHeader("Data Upload & Basic MS2LDA Parameters"),
                    dbc.CardBody(
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
                                    dbc.Input(id="n-iterations", type="number", value=10000),
                                ],
                                className="mb-3",
                                id="iterations-inputgroup",
                            ),
                            dbc.Tooltip(
                                "Number of LDA training iterations. Higher = more thorough training.",
                                target="iterations-tooltip",
                                placement="right",
                            ),
                        ]
                    ),
                ],
                style={"border": "1px solid #ccc", "padding": "10px", "marginBottom": "20px"},
            ),

            # Card 3: Spec2Vec Setup
            dbc.Card(
                [
                    dbc.CardHeader("Spec2Vec Setup"),
                    dbc.CardBody(
                        [
                            dcc.Markdown(
                                """
                                **Spec2Vec** is used to annotate motifs by comparing them against a pretrained model 
                                and library embeddings. You can download the necessary files here if you haven't already. 
                                If the files already exist, the process will skip them.
                                """
                            ),
                            dcc.Loading(
                                id="download-s2v-spinner",
                                type="circle",
                                children=[
                                    dbc.Button(
                                        "Download Spec2Vec Files",
                                        id="download-s2v-button",
                                        color="info",
                                        className="mt-3"
                                    ),
                                    html.Div(id="download-s2v-status", style={"marginTop": "10px"})
                                ]
                            ),
                            html.Br(),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("S2V Model Path", id="s2v-model-tooltip"),
                                    dbc.Input(
                                        id="s2v-model-path",
                                        type="text",
                                        value="../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_Spec2Vec_pos_CleanedLibraries.model",
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
                                    dbc.InputGroupText("S2V Library Embeddings", id="s2v-library-embeddings-tooltip"),
                                    dbc.Input(
                                        id="s2v-library-embeddings",
                                        type="text",
                                        value="../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy",
                                    ),
                                ],
                                className="mb-3",
                                id="s2v-library-embeddings-ig",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("S2V Library DB", id="s2v-library-db-tooltip"),
                                    dbc.Input(
                                        id="s2v-library-db",
                                        type="text",
                                        value="../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CombLibraries_spectra.db",
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
                        ]
                    ),
                ],
                style={"border": "1px solid #ccc", "padding": "10px", "marginBottom": "20px"},
            ),

            # Card 4: Advanced Settings
            dbc.Card(
                [
                    dbc.CardHeader("Advanced Settings"),
                    dbc.CardBody(
                        [
                            dcc.Markdown(
                                """
                                These parameters allow fine-grained control over MS2LDA preprocessing, 
                                convergence criteria, and model hyperparameters. Generally, 
                                you can leave them as defaults unless you want to experiment 
                                with more specialized behaviors or custom thresholding.
                                """
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
                                                    html.H6("Preprocessing"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("min_mz", id="prep-minmz-tooltip"),
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
                                                            dbc.InputGroupText("max_mz", id="prep-maxmz-tooltip"),
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
                                                            dbc.InputGroupText("max_frags", id="prep-maxfrags-tooltip"),
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
                                                            dbc.InputGroupText("min_frags", id="prep-minfrags-tooltip"),
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
                                                    html.H6("Convergence"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("step_size", id="conv-stepsz-tooltip"),
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
                                                            dbc.InputGroupText("window_size", id="conv-winsz-tooltip"),
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
                                                            dbc.InputGroupText("threshold", id="conv-thresh-tooltip"),
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
                                                            dbc.InputGroupText("criterium", id="ann-criterium-tooltip"),
                                                            dbc.Select(
                                                                id="ann-criterium",
                                                                options=[
                                                                    {"label": "best", "value": "best"},
                                                                    {"label": "biggest", "value": "biggest"},
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
                                                            dbc.InputGroupText("parallel", id="train-parallel-tooltip"),
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
                                                            dbc.InputGroupText("workers", id="train-workers-tooltip"),
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
                                                            dbc.InputGroupText("sig_digits", id="prep-sigdig-tooltip"),
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
                                                            dbc.InputGroupText("charge", id="dataset-charge-tooltip"),
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
                                                            dbc.InputGroupText("Run Name", id="dataset-name-tooltip"),
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
                                ]
                            ),
                        ]
                    ),
                ],
                style={"border": "1px solid #ccc", "padding": "10px", "marginBottom": "20px"},
            ),

            dbc.Card(
                [
                    dbc.CardHeader("Run Analysis"),
                    dbc.CardBody(
                        [
                            dcc.Markdown(
                                """
                                Once everything is configured, click **Run Analysis** 
                                to perform LDA on your spectra. Depending on the dataset 
                                size and iterations, this can take a while. Please wait until the 
                                progress indicator has finished to retrieve the results.
                                """
                            ),
                            dcc.Loading(
                                id="run-analysis-spinner",
                                type="circle",
                                children=[
                                    dbc.Button(
                                        "Run Analysis",
                                        id="run-button",
                                        color="primary",
                                    ),
                                    html.Div(id="run-status", style={"marginTop": "20px"})
                                ]
                            ),
                        ]
                    ),
                ],
                style={"border": "1px solid #ccc", "padding": "10px", "marginBottom": "20px"},
            ),
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
                        This tab allows you to load previously generated MS2LDA results (a compressed JSON file). 
                        Once loaded, you can explore them immediately in the subsequent tabs. 
                        This is useful if you’ve run an analysis before and want to revisit or share the results.
                        """
                    )
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
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
                style={"marginTop": "20px", "marginBottom": "20px"},
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
                            This tab displays your motifs in a ranked table based on how many documents meet 
                            the selected Probability and Overlap ranges. For each motif, we compute a `Degree` 
                            representing the number of documents where the motif’s doc-topic probability and overlap 
                            score both fall within the selected threshold ranges. We also report an `Average Doc-Topic Probability` 
                            and an `Average Overlap Score`. These averages are computed only over the documents that pass the filters, 
                            so they can be quite high if the motif strongly dominates the docs where it appears. 

                            Adjust the two RangeSliders below to narrow the doc-level thresholds on Probability and Overlap. 
                            _A motif remains in the table only if at least one document passes these filters_. Clicking on a motif row 
                            takes you to a detailed view of that motif.
                            """
                        )
                    ],
                    style={"marginTop": "20px", "marginBottom": "20px"},
                ),

                dbc.Row([
                    dbc.Col([
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
                                            value=[0, 1],
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
                                            value=[0, 1],
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
                        html.Div(id="motif-rankings-count", style={"marginTop": "10px"}),
                        dash_table.DataTable(
                            id='motif-rankings-table',
                            data=[],
                            columns=[],
                            sort_action='native',
                            filter_action='native',
                            page_size=20,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'minWidth': '150px', 'width': '200px', 'maxWidth': '400px',
                                'whiteSpace': 'normal',
                                'textAlign': 'left',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Motif'},
                                    'cursor': 'pointer',
                                    'textDecoration': 'underline',
                                    'color': 'blue'
                                },
                            ],
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                        ),
                        dbc.Button("Save to CSV", id="save-motifranking-csv", color="secondary", className="mt-2"),
                        dbc.Button("Save to JSON", id="save-motifranking-json", color="secondary", className="ms-2 mt-2"),
                        dcc.Download(id="download-motifranking-csv"),
                        dcc.Download(id="download-motifranking-json"),
                    ], width=12),
                ]),
            ]),
        ],
        style={"display": "none"},
    )
    return tab



def create_motif_details_tab():
    tab = html.Div(
        id="motif-details-tab-content",
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
                        """
                    )
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),

            # ----------------------------------------------------------------
            # 1. MOTIF DETAILS
            # ----------------------------------------------------------------
            html.Div([
                html.H4(id='motif-details-title'),

                html.Div([
                    html.H5("Spec2Vec Matching Results"),
                    dcc.Markdown(
                        """
                        The Spec2Vec matching results displayed here suggest chemical structures (SMILES strings) that closely match the selected motif. Spec2Vec calculates similarities by comparing motif pseudo-spectra against reference spectra from a known database. Matches shown here can help identify possible chemical identities or provide clues about structural characteristics represented by this motif.
                        """
                    ),
                    html.Div(id='motif-spec2vec-container', style={"marginTop": "10px"}),
                ], style={
                    "border": "1px dashed #ccc", "padding": "10px", "borderRadius": "5px", "marginBottom": "15px"
                }),

                html.Div([
                    html.H5("Optimised Motif Pseudo-Spectrum"),
                    dcc.Markdown(
                        """
                        This plot shows the optimised pseudo-spectrum, which refines the raw motif by retaining only fragments or losses consistently found among the best Spec2Vec matches. Because it's derived from verified library spectra, the optimised motif doesn't rely on the probability thresholds used in the original LDA motif extraction. You can use the toggle below to switch between displaying fragments or losses, depending on the feature type you want to explore.
                        """
                    ),
                    dbc.RadioItems(
                        id='optimised-motif-fragloss-toggle',
                        options=[
                            {"label": "Fragments Only", "value": "fragments"},
                            {"label": "Losses Only", "value": "losses"},
                        ],
                        value="fragments",
                        inline=True,
                    ),
                    html.Div(id='motif-optimized-spectrum-container', style={"marginTop": "10px"}),
                ], style={
                    "border": "1px dashed #ccc", "padding": "10px", "borderRadius": "5px", "marginBottom": "15px"
                }),

            ], style={
                "border": "1px dashed #999", "padding": "15px", "borderRadius": "5px", "marginBottom": "20px"
            }),

            # ----------------------------------------------------------------
            # 2. FEATURES IN MOTIFS
            # ----------------------------------------------------------------
            html.Div([
                html.H4("Features in Motifs"),

                html.Div([
                    html.H5("Spectra-Peaks Probability Filter"),
                    dcc.Markdown(
                        """
                        The slider below controls the minimum and maximum probability thresholds for selecting motif features (fragments and losses) identified by the LDA model. Setting a higher minimum value keeps only peaks strongly associated with the motif, providing a simpler representation. A lower minimum includes more peaks, potentially capturing additional detail but also introducing noise.
                        """
                    ),
                    dbc.Label("Spectra-Peaks Probability Filter:"),
                    dcc.RangeSlider(
                        id='probability-filter', min=0, max=1, step=0.01, value=[0, 1],
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'}, allowCross=False
                    ),
                    html.Div(id='probability-filter-display', style={"marginTop": "10px"}),
                ], style={
                    "border": "1px dashed #ccc", "padding": "10px", "borderRadius": "5px", "marginBottom": "10px"
                }),

                html.Div([
                    html.H5("Raw LDA Motif Pseudo-Spectrum (Filtered)"),
                    dcc.Markdown(
                        """
                        This plot visualizes the pseudo-spectrum of the raw motif after filtering peaks based on the selected probability thresholds. Peaks outside the defined range are removed, showing only the features most relevant according to the LDA model.
                        """
                    ),
                    html.Div(id='motif-raw-spectrum-container', style={"marginTop": "10px"}),
                ], style={
                    "border": "1px dashed #ccc", "padding": "10px", "borderRadius": "5px", "marginBottom": "10px"
                }),

                html.Div([
                    html.H5("Motif Features Table and Summary Plots"),
                    dcc.Markdown(
                        """
                        The table below lists the motif features (fragments and losses) that pass the probability filter, including their probabilities within the motif. Below it, a bar plot shows how frequently each feature appears **within the filtered set of documents** for this motif (i.e., documents whose doc-topic probability and overlap score both pass the current threshold ranges).
                        """
                    ),
                    html.Div(id='motif-features-container'),
                ], style={
                    "border": "1px dashed #ccc", "padding": "10px", "borderRadius": "5px", "marginBottom": "10px"
                }),

            ], style={
                "border": "1px dashed #999", "padding": "15px", "borderRadius": "5px", "marginBottom": "20px"
            }),

            # ----------------------------------------------------------------
            # 3. SPECTRA IN MOTIFS
            # ----------------------------------------------------------------
            html.Div([
                html.H4("Spectra in Motifs"),

                html.Div([
                    dcc.Markdown(
                        """
                        This section lists actual MS2 spectra from the dataset associated with the selected motif. Two adjustable sliders below control how these spectra are filtered. The Spectra-Motif Probability Filter determines the minimum and maximum probability a spectrum must have to be associated with this motif, while the Overlap Score Filter sets how closely the spectrum’s peaks must align with the motif features.

                        Select a spectrum from the table or use the Next/Previous buttons to browse through them. Each selection updates the spectrum plot below, highlighting peaks matching features from the motif. This plot helps verify if the motif truly represents meaningful chemical information by directly comparing it against real experimental data.
                        """
                    ),

                    dbc.Label("Spectra-Motif Probability Filter:"),
                    dcc.RangeSlider(
                        id='doc-topic-filter', min=0, max=1, step=0.01, value=[0, 1],
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'}, allowCross=False
                    ),
                    html.Div(id='doc-topic-filter-display', style={"marginTop": "10px"}),

                    dbc.Label("Overlap Score Filter:"),
                    dcc.RangeSlider(
                        id='overlap-filter', min=0, max=1, step=0.01, value=[0, 1],
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'}, allowCross=False
                    ),
                    html.Div(id='overlap-filter-display', style={"marginTop": "10px"}),

                    html.Div(id='motif-documents-container'),

                    dash_table.DataTable(
                        id='spectra-table', data=[], columns=[],
                        style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left'},
                        page_size=10, row_selectable='single', selected_rows=[0], hidden_columns=["SpecIndex"]
                    ),
                    html.Div(id='spectrum-plot'),

                    html.Div([
                        dbc.Button('Previous', id='prev-spectrum', n_clicks=0, color="info"),
                        dbc.Button('Next', id='next-spectrum', n_clicks=0, className='ms-2', color="info"),
                    ], className='mt-3'),
                ], style={
                    "border": "1px dashed #ccc", "padding": "10px", "borderRadius": "5px", "marginBottom": "15px"
                }),

            ], style={
                "border": "1px dashed #999", "padding": "15px", "borderRadius": "5px", "marginBottom": "20px"
            }),

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
                style={"marginTop": "20px", "marginBottom": "20px"},
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
                    {"name": "User AutoAnno", "id": "user_auto_annotation"},
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


def create_spectra_search_tab():
    tab = html.Div(
        id="search-spectra-tab-content",
        style={"display": "none"},
        children=[
            html.H3("Search by Fragment/Loss or Parent Mass"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Fragment or Loss Contains:"),
                    dbc.Input(
                        id="spectra-search-fragloss-input",
                        type="text",
                        placeholder="e.g. frag@150 or loss@40",
                    ),
                    # ADD THIS TOOLTIP:
                    dbc.Tooltip(
                        "Enter a partial fragment (e.g., 'frag@150.1') or loss (e.g., 'loss@40.2'). "
                        "The search is case-insensitive and matches any part of the feature string.",
                        target="spectra-search-fragloss-input",
                    ),
                ], width=4),

                dbc.Col([
                    dbc.Label("Parent Mass Range:"),
                    dcc.RangeSlider(
                        id="spectra-search-parentmass-slider",
                        min=0,
                        max=2000,
                        step=1,
                        value=[0, 2000],
                        marks={0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"},
                        allowCross=False,
                    ),
                    html.Div(id="spectra-search-parentmass-slider-display", style={"marginTop": "10px"}),
                ], width=8),
            ], style={"marginTop": "20px"}),

            # ADD STATUS MESSAGE DIV HERE
            html.Div(
                id="spectra-search-status-message",
                style={"marginTop": "10px", "fontStyle": "italic", "color": "#555"},
            ),

            dash_table.DataTable(
                id="spectra-search-results-table",
                columns=[
                    {"name": "Spectrum ID",   "id": "spec_id"},
                    {"name": "Parent Mass",  "id": "parent_mass"},
                    {"name": "Feature ID",   "id": "feature_id"},
                    {"name": "Fragments",    "id": "fragments"},
                    {"name": "Losses",       "id": "losses"},
                ],
                data=[],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "whiteSpace": "normal"},
                row_selectable="single",
            ),

            # Add a container for the selected spectrum details
            dcc.Store(id="search-tab-selected-spectrum-details-store"),
            dcc.Store(id="search-tab-selected-motif-id-for-plot-store"),

            # Add a container for the spectrum details
            html.Div(
                id="search-tab-spectrum-details-container",
                style={"marginTop": "20px", "display": "none"},
                children=[
                    html.H4("Spectrum Details"),

                    # Associated Motifs
                    html.Div([
                        html.H5("Associated Motifs"),
                        html.Div(
                            id="search-tab-associated-motifs-list",
                            style={"marginTop": "10px"},
                        ),
                    ], style={
                        "border": "1px dashed #ccc",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginBottom": "15px"
                    }),

                    # Spectrum Plot
                    html.Div([
                        html.H5("Spectrum Plot"),
                        html.Div(
                            id="search-tab-spectrum-plot-container",
                            style={"marginTop": "10px"},
                        ),
                    ], style={
                        "border": "1px dashed #ccc",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginBottom": "15px"
                    }),
                ],
            ),
        ],
    )
    return tab
