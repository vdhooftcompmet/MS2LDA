# Quick Start Demo

Try MS2LDA with example data in just a few minutes! This guide shows how to test MS2LDA in a clean environment.

## Step 1: Create a Virtual Environment

Choose your preferred method to create a new virtual environment (or use an existing one):

### Option A: Using Conda
```bash
conda create -n ms2lda_demo python=3.11 -y
conda activate ms2lda_demo
```

### Option B: Using Poetry
```bash
poetry new ms2lda_demo
cd ms2lda_demo
```
**Note**: Poetry users need to prefix commands with `poetry run` (e.g., `poetry run ms2lda ...`)

### Option C: Using virtualenv
```bash
python -m venv ms2lda_venv
source ms2lda_venv/bin/activate  # On Windows: ms2lda_venv\Scripts\activate
```

## Step 2: Install MS2LDA and Run Demo

Once you have your environment activated, the remaining steps are the same:

```bash
# Install MS2LDA from PyPI
pip install ms2lda  # Poetry users: use 'poetry add ms2lda' instead

# Download and extract example datasets
wget https://zenodo.org/records/15857387/files/datasets.zip?download=1 -O datasets.zip
unzip datasets.zip
rm datasets.zip

# Run MS2LDA on example fungal dataset
ms2lda --dataset datasets/Case_Study_Fungal_dataset.mgf \
       --n-motifs 200 \
       --n-iterations 10000 \
       --output-folder ms2lda_demo_results

# Note: This uses default parameters. For the exact parameters used in our paper,
# see: https://github.com/vdhooftcompmet/MS2LDA/blob/main/notebooks/Paper_results/CaseStudy_Mushrooms_run.ipynb
# For analysis examples, see: https://github.com/vdhooftcompmet/MS2LDA/blob/main/notebooks/Paper_results/CaseStudy_Mushrooms_analysis.ipynb

# Launch the visualization app to explore results
ms2lda-viz
# Open http://localhost:8050 in your browser
```

## What's Created

The analysis will create a folder `ms2lda_demo_results/` with:
- Discovered motifs (Mass2Motifs)
- Document-motif distributions
- Interactive visualizations
- Automated annotations via Spec2Vec

## Next Steps

- Explore the web interface at http://localhost:8050
- Try different datasets from the `datasets/` folder
- Adjust parameters like `--n-motifs` and `--n-iterations`
- See [Command Line Tool Guide](README_CLI.md) for more options