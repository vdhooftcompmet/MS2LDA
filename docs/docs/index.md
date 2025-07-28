![MS2LDA Logo](./figures/MS2LDA_LOGO_white.jpg)

# Welcome to the MS2LDA Documentation

MS2LDA (Mass Spectrometry–Latent Dirichlet Allocation) is a framework that brings **topic modeling** to the world of **tandem mass spectrometry (MS/MS)**. It helps identify recurring fragmentation patterns — known as **Mass2Motifs** — that represent conserved molecular **substructures** across complex spectra.

---

## What is MS2LDA?

Traditional mass spectrometry analysis depends on spectral libraries or manual curation. MS2LDA offers a **machine learning based, data driven, and unsupervised alternative** that:

- Detects latent fragmentation motifs across large datasets.
- Aids structural elucidation of unknown compounds.
- Bridges mass spectrometry and cheminformatics.

MS2LDA applies **Latent Dirichlet Allocation (LDA)**, originally developed for text analysis, to fragment ions and neutral losses. This allows the discovery of statistically significant patterns that often reflect chemical substructures 🔍

---

## Key Features

- 🧠 **Unsupervised learning** of Mass2Motifs at unprecedented speed
- 🧬 **Automated Mass2Motif Annotation Guidance (MAG)** with Spec2Vec
- 🔗 **Integration** with MassQL-searchable MotifDB
- 📈 **Visualization app** for interactive exploration of Mass2Motifs
- 💻 **Command-line access** for advanced users

---

## Documentation Sections

This site provides everything you need to get started:

- [**Getting Started**](./home/quick_start.md): Installation, quickstart, and basic usage
- [**User Guide**](./guide/overview): Workflow explanation and parameter tuning
- [**Modules Reference**](./api/): All available classes and functions
- [**Examples & Tutorials**](./examples/): Practical use cases and annotated datasets

---

## Developers & Contributors

MS2LDA is developed by a team led by **Rosina Torres Ortega**, **Jonas Dietrich**, and **Joe Wandy**, under the supervision of **Justin J.J. van der Hooft** at Wageningen University & Research.


📚 MS2LDA builds on the original work published in:

**van der Hooft et al. PNAS, 2016** → [https://doi.org/10.1073/pnas.1608041113](https://doi.org/10.1073/pnas.1608041113)

As well as MotifDB:

**Rogers et al. Faraday Discussions, 2019** → [https://doi.org/10.1039/C8FD00235E](https://doi.org/10.1039/C8FD00235E)

📝 For methodology details and recent updates, please read our preprint:  
**Torres Ortega et al. bioRxiv, 2025** → [https://doi.org/10.1101/2025.06.19.659491](https://doi.org/10.1101/2025.06.19.659491)


Ongoing development continues in collaboration with the broader metabolomics and computational biology community. We welcome feedback, issues, and pull requests on our [GitHub repository](https://github.com/vdhooftcompmet/MS2LDA).  

Questions? Open an issue or contact the development team directly 🤝

---
## Acknowledgments

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <td style="border: none; width: 50%; vertical-align: top;">
      <img src="./figures/CompMetabolomics_logo.jpg" alt="CompMetabolomics Logo" style="height: 100px;">
    </td>
    <td style="border: none; width: 50%; vertical-align: top;">
      <img src="./figures/WUR_logo.jpg" alt="WUR Logo" style="height: 100px;">
    </td>
  </tr>
  <tr>
    <td style="border: none; text-align: justify; vertical-align: top;">
      This work was carried out by the van der Hooft Computational Metabolomics Group.
    </td>
    <td style="border: none; text-align: justify; vertical-align: top;">
      This work was supported by Wageningen University & Research.
    </td>
  </tr>
</table>
