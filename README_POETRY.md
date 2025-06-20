![header](App/assets/MS2LDA_LOGO_white.jpg)
![Maintainer](https://img.shields.io/badge/maintainer-Rosina_Torres_Ortega-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Jonas_Dietrich-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Joe_Wandy-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12625409.svg)](https://doi.org/10.5281/zenodo.11394248)

**MS2LDA** is an advanced tool designed for unsupervised substructure discovery in mass spectrometry data, utilizing topic modeling and providing automated annotation of discovered motifs. This tool significantly enhances the capabilities described in the [original MS2LDA paper](https://www.pnas.org/doi/abs/10.1073/pnas.1608041113) (2016), offering users an integrated workflow with improved usability, detailed visualizations, and a searchable motif database (MotifDB).

Mass spectrometry fragmentation patterns hold abundant structural information vital for analytical chemistry, natural product research, and food safety assessments. However, interpreting this data remains challenging, and only a fraction of available information is traditionally utilized. MS2LDA addresses this by identifying recurring substructures (motifs) across spectral datasets without relying on prior compound identification, thus accelerating structure elucidation and analysis.

---

# MS2LDA Installation and Usage with Poetry

This guide provides instructions for setting up and using MS2LDA with Poetry for dependency management.

## Installation

Poetry provides modern Python dependency management and is particularly suited for development or customized installations.

**Step 1: Install Poetry**
Follow the [official Poetry documentation](https://python-poetry.org/docs/#installation) for installation instructions suitable to your operating system.

**Step 2: Clone Repository and Install**
Clone and install dependencies:

```bash
git clone https://github.com/vdhooftcompmet/MS2LDA.git
cd MS2LDA
poetry install
```

**Important Notes for RDKit with Poetry:**

* RDKit installation via Poetry typically uses pre-built binaries, which work well for common systems. However, for certain hardware or OS configurations (especially ARM-based Macs or less common Linux distributions), pre-built binaries might not be available.
* In such cases, we recommend installing RDKit separately using Conda, then installing MS2LDA without the RDKit binary:

```bash
poetry install --extras lite --no-binary rdkit
```

**Step 3: Download Required Data and Models**
Run the provided script to download all necessary files:

```bash
poetry run ./run_analysis.sh --only-download
```

Alternatively, you can download models and datasets from [Zenodo](https://zenodo.org/records/15003249). Extract and place them into the appropriate directories as indicated in the Zenodo repository.

## Running MS2LDA via Command-Line Interface (CLI)

MS2LDA includes a powerful CLI enabling batch analysis and scripting. Before running any analysis, it's recommended to download all necessary files:

```bash
poetry run ./run_analysis.sh --only-download
```

Run basic analysis:

```bash
poetry run ./run_analysis.sh --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results
```

Alternatively, you can directly run analyses via:

```bash
poetry run ms2lda --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results
```

## Running the Interactive Dash Application

The Dash application provides a comprehensive, user-friendly graphical interface for exploring mass spectrometry datasets, viewing motifs, screening substructures, and visualizing results interactively.

```bash
poetry run ./run_ms2ldaviz.sh
```

For Windows users:
```bash
poetry run run_ms2ldaviz.bat
```

Alternatively, you can directly run the visualization via:

```bash
poetry run ms2lda-viz
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

To launch notebooks with Poetry:

```bash
poetry install --extras notebook
poetry run jupyter lab
```

---

## Documentation

For more comprehensive guidance, refer to our complete [MS2LDA Documentation](https://ms2lda.org/), providing detailed instructions, FAQs, and additional resources.

---

## Citing MS2LDA

Please cite our work if you use MS2LDA in your research. *(Citation details coming soon.)*

---

## Our Research Group

[![GitHub Logo](https://github.com/vdhooftcompmet/group-website/blob/main/website/custom/logo/logo.png?raw=true)](https://vdhooftcompmet.github.io)
[![Github Logo](App/assets/WUR_RGB_standard_2021.png?raw=true)](https://www.wur.nl/en.htm)

---
