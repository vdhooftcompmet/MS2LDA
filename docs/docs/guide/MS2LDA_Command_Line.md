# MS2LDA Command-Line Interface Guide 🖥️

This document describes how to run **MS2LDA** from the command line, including available commands, flags, parameters files, etc.

---

The MS2LDA repository includes convenient scripts that simplify its execution.
If you have not cloned the repository and created a conda enviroment [(Conda Website)](http://conda.io/), please go to [**Getting Started**](./home/quick_start.md), afterwards you will find inside the MS2LDA folder the following scripts:

- **`run_analysis.sh`** (Linux/macOS)  
- **`run_analysis.bat`** (Windows)

These scripts will:

1. Configure the Python environment (set `PYTHONPATH`, activate conda, etc.)  
2. Invoke the main script `ms2lda_runfull.py` with your arguments  
3. Handle any platform‑specific quirks

## 1. Quick Help

To see a list of commands and global options, run:

```bash
./run_analysis.sh --help
```
Usage: run_ms2lda.py [OPTIONS]

Options:
  --input PATH           Path to input spectra file (.mgf, .mzML, .msp)
  --output DIR           Directory to store results
  --params JSON          JSON file with module parameters
  --n_topics INTEGER     Number of Mass2Motifs to infer
  --n_iterations INTEGER Number of LDA training iterations
  --alpha FLOAT          LDA alpha hyperparameter
  --beta FLOAT           LDA beta hyperparameter
  --log_level LEVEL      Logging level (DEBUG, INFO, WARN, ERROR)
  -h, --help             Show this message and exit

## 2. Download Required Datasets

```bash
# For Linux/macOS:
./run_analysis.sh --only-download

# For Windows:
run_analysis.bat --only-download
```

## 3. Minimal Run

```bash
./run_analysis.sh --dataset <input_file> --n-motifs <number> --n-iterations <number> --output-folder <folder>
```
**Explanation**

| Flag             | Type   | Default | Description                                       |
| ---------------- | ------ | ------- | ------------------------------------------------- |
| `--dataset`      | string | —       | Path to input spectra file                        |
| `--n-motifs`     | int    | —       | Preferred number of motifs                        |
| `--n-iterations` | int    | —       | Number of iterations                              |
| `--n_topics`     | int    | —       | Number of topics (Mass2Motifs) to infer           |

## 4. Logs & Output Structure
After a successful run, your results/ directory will contain:

```text
results/
├─ motif_figures/          # Folder with individual motif visualizations (PNG files)
├─ motifs/                 # Folder with each inferred Mass2Motif
├─ motifset.json           # Discovered Mass2Motifs in JSON format
├─ motifset_optimized.json # Optimized Mass2Motifs in JSON format
├─ doc2spec_map.pkl        # Pickled mapping of documents to original spectra
├─ convergence_curve.png   # Training convergence plot
├─ network.graphml         # Molecular network export (GraphML)
├─ ms2lda.bin              # Binary dump of the trained LDA model
└─ ms2lda_viz.json.gz      # Compressed results for the MS2LDAViz web app
```

For advanced information on how to tune your MS2LDA analysis and the usage of Jupyter Notebooks please go to: [GitHub repository](https://github.com/vdhooftcompmet/MS2LDA/blob/main/README_CLI.md).  