![header](App/assets/MS2LDA_LOGO_white.jpg)
![Maintainer](https://img.shields.io/badge/maintainer-Rosina_Torres_Ortega-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Jonas_Dietrich-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Joe_Wandy-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12625409.svg)](https://doi.org/10.5281/zenodo.11394248)

**MS2LDA** is an advanced tool designed for unsupervised substructure discovery in mass spectrometry data, utilizing topic modeling and providing automated annotation of discovered motifs. This tool significantly enhances the capabilities described in the [original MS2LDA paper](https://www.pnas.org/doi/abs/10.1073/pnas.1608041113) (2016), offering users an integrated workflow with improved usability, detailed visualizations, and a searchable motif database (MotifDB).

Mass spectrometry fragmentation patterns hold abundant structural information vital for analytical chemistry, natural product research, and food safety assessments. However, interpreting this data remains challenging, and only a fraction of available information is traditionally utilized. MS2LDA addresses this by identifying recurring substructures (motifs) across spectral datasets without relying on prior compound identification, thus accelerating structure elucidation and analysis.

---

## Installation

You can set up MS2LDA using either **Conda** (recommended for ease and reliability) or **Poetry** (preferred by developers or users needing finer control over dependencies).

### Option A: Installation via Conda (Recommended)

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
Download necessary models and datasets from [Zenodo](https://zenodo.org/records/15003249). Extract and place them into the appropriate directories as indicated in the Zenodo repository.

---

### Option B: Installation via Poetry (Alternative)

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

---

## Running MS2LDA via Command-Line Interface (CLI)

MS2LDA includes a powerful CLI enabling batch analysis and scripting. You can execute analyses using the included runner scripts (recommended for simplicity) or via Poetry's installed commands.

**Using the runner scripts:** (Linux/MacOS)

Run basic analysis:

```bash
./run_analysis.sh --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results
```

Run analysis and automatically download Spec2Vec data if not present:

```bash
./run_analysis.sh --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results --download-spec2vec
```

Use a configuration file for advanced settings:

```bash
./run_analysis.sh --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder test_results --config default_config.json
```

**Using Poetry command (Editable install):**

If you've installed MS2LDA with Poetry, you can directly run analyses via:

```bash
poetry run ms2lda --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results
```

---

## Running the Interactive Dash Application

The Dash application provides a comprehensive, user-friendly graphical interface for exploring mass spectrometry datasets, viewing motifs, screening substructures, and visualizing results interactively.

Activate your environment first:

* **Conda:**

  ```bash
  conda activate MS2LDA_v2
  ```
* **Poetry:**

  ```bash
  poetry shell
  ```

Then, launch the Dash app using:

* **Windows:**

  ```bash
  run_ms2ldaviz.bat
  ```

* **Linux/MacOS:**

  ```bash
  ./run_ms2ldaviz.sh
  ```

* **Or via Poetry (Editable install):**

  ```bash
  poetry run ms2lda-viz
  ```

Navigate your web browser to [http://localhost:8050](http://localhost:8050) to start exploring.

---

## Jupyter Notebooks

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

For more comprehensive guidance, refer to our complete [MS2LDA Documentation](https://vdhooftcompmet.github.io/MS2LDA/), providing detailed instructions, FAQs, and additional resources.

---

## Citing MS2LDA

Please cite our work if you use MS2LDA in your research. *(Citation details coming soon.)*

---

## Our Research Group

[![GitHub Logo](https://github.com/vdhooftcompmet/group-website/blob/main/website/custom/logo/logo.png?raw=true)](https://vdhooftcompmet.github.io)
[![Github Logo](App/assets/WUR_RGB_standard_2021.png?raw=true)](https://www.wur.nl/en.htm)

---
