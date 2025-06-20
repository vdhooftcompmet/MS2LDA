
# MS2LDA Installation and Usage with Conda

This guide provides instructions for setting up and using MS2LDA with Conda environment management.

## Installation

**Step 1: Install Conda**
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or the complete [Anaconda](https://www.anaconda.com/products/distribution) distribution.

**Step 2: Clone the Repository**
Clone the MS2LDA repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/vdhooftcompmet/MS2LDA.git
cd MS2LDA
```

**Step 3: Create and Activate Environment**
Use Conda to create and activate the required environment from the provided configuration file:

```bash
conda env create -f MS2LDA_environment.yml
conda activate MS2LDA_v2
```

**Step 4: Download Required Data and Models**
Run the provided script to download all necessary files:

```bash
./run_analysis.sh --only-download
```

Alternatively, you can download models and datasets from [Zenodo](https://zenodo.org/records/15003249). Extract and place them into the appropriate directories as indicated in the Zenodo repository.

## Running MS2LDA via Command-Line Interface (CLI)

MS2LDA includes a powerful CLI enabling batch analysis and scripting. Before running any analysis, it's recommended to download all necessary files:

```bash
./run_analysis.sh --only-download
```

Run basic analysis:

```bash
conda activate MS2LDA_v2
./run_analysis.sh --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results
```

Use a configuration file for advanced settings:

```bash
./run_analysis.sh --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder test_results --config default_config.json
```

## Running the Interactive Dash Application

The Dash application provides a comprehensive, user-friendly graphical interface for exploring mass spectrometry datasets, viewing motifs, screening substructures, and visualizing results interactively.

```bash
conda activate MS2LDA_v2
./run_ms2ldaviz.sh
```

For Windows users:
```bash
conda activate MS2LDA_v2
run_ms2ldaviz.bat
```

Navigate your web browser to [http://localhost:8050](http://localhost:8050) to start exploring.

If you are hosting the dashboard on a resource-limited server and want to disable
the "Run Analysis" tab, set the environment variable `ENABLE_RUN_ANALYSIS=0`
before launching the app. This hides the analysis tab but still allows loading
existing results locally.

## Using Jupyter Notebooks

Included in the repository are several Jupyter notebooks organized in the `notebooks` directory, providing tutorials, conversion utilities, detailed examples, and analyses replication from previous publications.

Key notebooks folders:

* **MotifSets**: Tools for converting motif formats.
* **Paper\_Results**: Code and data to reproduce published results.
* **Spec2Vec**: Reference libraries and embedding models.
* **Tutorial**: Interactive introductory tutorials and example workflows.
