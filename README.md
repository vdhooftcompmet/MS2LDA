![header](App/assets/MS2LDA_LOGO_white.jpg)
![Maintainer](https://img.shields.io/badge/maintainer-Rosina_Torres_Ortega-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Jonas_Dietrich-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Joe_Wandy-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12625409.svg)](https://doi.org/10.5281/zenodo.11394248)

Structure elucidation is a major bottleneck in analytical chemistry and its progress influences various research fields like natural product discovery and food safety. Mass spectrometry is one of the most used instruments for structure elucidation and identification due to its information-rich fragmentation patterns. To date, only a fraction of this information is understood and can be related to structural features. For molecular properties such as bioactivity, substructures are key components. This implementation of MS2LDA is based on the original version published in 2017 and offers automated annotation, visualization tools, and a database for discovered motifs (MotifDB).

## Installation

### Step 1: Install Conda

If you don't already have Conda installed, download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution). You can choose either Miniconda or Anaconda depending on your preference.

### Step 2: Clone the MS2LDA repository

```bash
git clone https://github.com/vdhooftcompmet/MS2LDA.git
```

### Step 3: Set up the Conda environment

Navigate to the project directory and create the environment:

```bash
cd MS2LDA
conda env create -f MS2LDA_environment.yml
conda activate MS2LDA_v2
```

### Step 4: Download models and datasets

Download all required models and datasets from [Zenodo](https://zenodo.org/records/12625409).

## Running the Dash Application

The MS2LDA Dash application provides an interactive platform for analyzing mass spectrometry data, visualizing motifs, and screening results. It includes tabs for running new analyses, loading existing results, viewing interactive networks, exploring detailed motif rankings, inspecting individual motif features and documents, and screening motifs against reference datasets.

To run the Dash app locally, ensure you activate the Conda environment first:

```bash
conda activate MS2LDA_v2
```

- On **Windows**, execute the batch script:

```bash
run_app.bat
```

- On **Linux/MacOS**, execute the shell script:

```bash
./run_app.sh
```

The Dash application will be accessible at `http://localhost:8051`.

## Notebooks

The repository includes a `notebooks` folder containing various Jupyter notebooks organized into subfolders:

- **MotifSets**: Utilities for converting old motif formats to new.
- **Paper_results**: Notebooks and data used for reproducing analyses and results described in MS2LDA publications.
- `Spec2Vec` includes notebooks for creating reference spectral libraries and retraining Spec2Vec embeddings.
- `Tutorial` provides introductory materials and example usage.

## Documentation

Full documentation is available at [MS2LDA Documentation](https://vdhooftcompmet.github.io/MS2LDA/).

## Cite us

Not published yet.

## More information about our research group

[![GitHub Logo](https://github.com/vdhooftcompmet/group-website/blob/main/website/custom/logo/logo.png?raw=true)](https://vdhooftcompmet.github.io)
[![Github Logo](App/assets/WUR_RGB_standard_2021.png?raw=true)](https://www.wur.nl/en.htm)

