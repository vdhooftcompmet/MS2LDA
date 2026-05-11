# MSn Library Analysis

Download the MSn data archive from Zenodo. Place these files in this folder:

- `Corinna_Library_filtered_positive.mgf`
- `CaseStudy_Corinna_Library_filtered_1000motifs_output_100625/`

Download the MS2LDA annotation models with:

```bash
ms2lda --only-download
```

Run notebooks from `notebooks/Paper_results` in this order:

1. `MSn_filtering.ipynb` creates `Corinna_Library_filtered_positive.mgf` from `merged_and_cleaned_libraries_1.mgf`. Skip this if the filtered MGF was downloaded directly. To rerun it, also download `merged_and_cleaned_libraries_1.mgf` and update the notebook input path to its location.
2. `Benchmark_MAG_MSn.ipynb` runs MS2LDA on `Corinna_Library_filtered_positive.mgf` and writes `CaseStudy_Corinna_Library_filtered_1000motifs_output_100625/`. Skip this if the output folder was downloaded directly.
3. `Analysis_MSnLib.ipynb` loads `CaseStudy_Corinna_Library_filtered_1000motifs_output_100625/` for Figure 2 / MSn analysis.
