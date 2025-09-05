# MS2LDAViz — Interactive Web Interface 🌐

MS2LDAViz lets you run analyses and explore your results in your browser, with live controls and rich visualizations.

---

## 1. Launching the App

The MS2LDA repository includes convenient scripts that allow the easy access to the Viz App. If you have not cloned the repository and created a conda enviroment, please go to [**Getting Started**](./home/quick_start.md), afterwards you will find inside the MS2LDA folder the following scripts:

- **`run_ms2ldaviz.sh`** (Linux/macOS)  
- **`run_ms2ldaviz.bat`** (Windows)

In order to use the scripts please type:

```bash
# For Linux/macOS
./run_ms2ldaviz.sh

# For Windows 
./run_ms2ldaviz.bat
```

Aftewards you will be redirected to the following [website](http://127.0.0.1:8000/….):
![Website page](../figures/MS2LDA_site_1.JPG)

Here you can see the following tabs:

- **Run Analysis**: Upload your MS/MS spectra, set core parameters, and start a new MS2LDA run. When it finishes, you can navigate the other tabs.
- **Load Results**: Open a previous MS2LDA run. This lets you explore without re-running the model.
- **Motif Rankings**: Browse all Mass2Motifs ranked by probability and overlap thresholds, or search for specific motifs using MassQL queries.
- **Motif Details**: See suggested chemical structures/annotations, its fragment and neutral-loss composition, and the associated spectra.
- **Spectra Search**: Find individual spectra by parent mass or fragment/loss values, and check their spectra and associated motifs.
- **View Network**: Explore an interactive network of optimized motifs, where each M2M is displayed as a node connected to its fragments ands losses.
- **Motif Search**: Perform motif-motif searches against reference motifs in MotifDB.