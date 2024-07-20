## Screening

Found motifs are mainly used in two ways. 
First, because you want to know what molecular substructures are present in your dataset. 
Therefore you run MS2LDA on an unknown dataset and explore and evaluate the automatic generated annotations.
Second, because you want to know if a certain substructure is present in a new dataset.
For latter reason we implemented a screening procedure that searches for substructure matches in an unknown dataset and tries to help annotate them as well.

## How does the screening work?

The screening process classifies spectra into four levels of confidence with the first one being the most confidence that the matching spectra contains the molecular substructure the motif represents.
Three criteria a check for the classification:
1. Number of matched peaks between spectrum and motif
2. Intensity of the matched fragments and losses in the motif
3. Trend of the matched feature intensities

In the following the four criteria needed for the classes A, B, C, and D are explained.

| Class | n features | cumulative feature intensity | Pearson for intensity trend |
|-------|------------|------------------------------|-----------------------------|
| A     | >= 3       | >= 0.9                       | >= 0.8                      |
| B     | == 2       | >= 0.9                       | >= 0.8                      |
| C     | >= 2       | >= 0.5                       | >= 0.6                      |
| D     | == 1       | >= 0.9                       | -                           |

## How does it help with the annotation of found molecules?

To assist with the annotation of potentially unknown new molecules found by screening the motif must be annotated with a SMILES. This allows to calculate the exact mass of the molecule, which will then be compared to the accurate precursor mass of the found molecues in the screening process.
The difference of the masses is possibly an addition or subtraction of a sidechain of the molecule. The molecular formula of the sidechain is calculate using [msbuddy](https://github.com/Philipbear/msbuddy). Expert knowledge and potentially NMR is needed for a full structure elucidation, but this approach can already help to get an idea of the structure.
OBVIOUSLY, it can also be the case that the found analog shares only a small part of the motif annotated molecule and therefore msbuddy cannot find any molecular subformula or a wrong one. The suggested annotations should always taken with caution.
