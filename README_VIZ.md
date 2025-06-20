
# MS2LDAViz: Visualization Application

MS2LDAViz is a web-based visualization application that allows you to explore and analyze MS2LDA results. This guide explains how to start and use the application.

## Starting MS2LDAViz

MS2LDA provides convenient scripts to start the visualization application:

- **For Linux/macOS users**: Use the `run_ms2ldaviz.sh` script
  ```bash
  ./run_ms2ldaviz.sh
  ```

- **For Windows users**: Use the `run_ms2ldaviz.bat` script
  ```
  run_ms2ldaviz.bat
  ```

After running the script, the application will start a local web server. Open your web browser and navigate to:
```
http://127.0.0.1:8050/
```

## MS2LDAViz Interface

The MS2LDAViz application provides several tabs for different functionalities:

### 1. Run Analysis
This tab allows you to run MS2LDA analysis directly from the web interface. You can:
- Upload MS data files
- Set analysis parameters
- Start the analysis process
- Monitor progress

### 2. Load Results
Use this tab to load previously generated MS2LDA results. You can:
- Select a results folder
- Load and explore the results

### 3. Motif Rankings
This tab displays a ranked list of discovered motifs. You can:
- Sort motifs by various metrics
- Filter motifs based on criteria
- Select motifs for detailed examination

### 4. Motif Details
This tab provides detailed information about selected motifs. You can:
- View fragment and loss patterns
- Examine mass spectra associated with the motif
- See structural annotations (if available)

### 5. Spectra Search
This tab allows you to search for specific spectra in your dataset. You can:
- Search by spectrum ID
- Filter spectra by various criteria
- Examine individual spectra in detail

### 6. View Network
This tab displays a network visualization of motifs and spectra. You can:
- Explore relationships between motifs
- Identify clusters of related spectra
- Interact with the network visualization

### 7. Motif Search
This tab allows you to search for specific motifs in MotifDB. You can:
- Search by motif characteristics
- Compare your motifs with reference motifs
- Identify potential structural annotations

## Tips for Using MS2LDAViz

- **Browser Compatibility**: MS2LDAViz works best with modern browsers like Chrome, Firefox, or Edge.
- **Data Loading**: Large datasets may take some time to load. Be patient during the initial loading process.
- **Interactive Visualizations**: Most visualizations in MS2LDAViz are interactive. Try clicking, hovering, and dragging elements to discover additional functionality.
- **Exporting Results**: Many visualizations and results can be exported for use in publications or further analysis.
