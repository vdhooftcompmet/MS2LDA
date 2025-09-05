# Getting Started

Welcome to MS2LDA! This quick guide helps you take your first steps into discovering interesting substructures in your data. So, let us install and use the tool, whether you're working through the web interface or the command line.

---

## Inputs & Outputs 📥📤

### Input
MS2LDA expects **preprocessed MS/MS data**, typically in:

- `.mgf`  (Mascot Generic Format)
- `.mzML` (via conversion or direct input)
- `.msp`  (NIST-style spectrum libraries)

### Output
After processing, MS2LDA provides:

- **Mass2Motifs** (discovered fragmentation patterns)
- **Spectra-motif loadings**
- Optional annotations via: **MotifDB** or **MAG**

Output formats:

- CSV tables
- JSON (for advanced integration)
- Visualizations (interactive in web app)

---

## Installation ⚙️

MS2LDA works on macOS, Linux, and Windows (Anaconda Prompt or Windows Subsystem for Linux).

**Note:** These steps assume you have [Conda](http://conda.io/) installed. On Windows, use the Anaconda Prompt or WSL.

```bash
# Clone the repository
git clone https://github.com/vdhooftcompmet/MS2LDA.git

# Load MS2LDA directory
cd MS2LDA
```
Now we will create the Conda environment using the provided **YAML configuration file (.yml)** included in the repository that specifies all the required Python packages and dependencies to run MS2LDA.

```bash
# Create and activate the environment 
conda env create -f MS2LDA_environment.yml

conda activate MS2LDA_v2
```

## Viz App vs Command-Line

For detailed information on how to run your analysis through:

- MS2LDAViz app please check: [MS2LDAViz App guide](../guide/MS2LDAViz_App.md)
- Commnad-line please check: [MS2LDA Command Line](../guide/MS2LDA_Command_Line.md)


