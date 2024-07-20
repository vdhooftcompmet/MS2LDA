# <img align="left" width="50" height="50" src="mask.png"> Spectral Masking

## Definition
Masking describes the process in language processing where we ignore some parts of the input.

## The Nature of Motifs
When applying a topic modeling algorithm to a set of spectra, the algorithm will group fragments and losses from different spectra that are the same or related because they happen to be often in the same spectra as another fragment that is the same across many spectra. In a huge and homogenous dataset, where every substructure has a variety of substituents and the structural diversity across the entire dataset is also very high, it should be possible to retrieve substructures by library matching paired with maximum common substructure (mcs) algorithms. 
In the real world, the datasets normally are dominated by a certain structure type, which also impacts the found motifs in a way that the fragments and losses of predominant structures can be found in most motifs even if not directly related. 
When using spectral library matching for motifs many times two structure types are present in the database hits in orbitrary order. The two different structure types are so different that they cannot be used for a mcs analysis. Structural based similarity search to seperate the two spectra can work to a certain extend, but will fail if the common substructure is small compared to the molecule size, ignoring the difficulty to select a correct fingerprint. The two different structure in our example are both partially represented by motif and makes it difficult to futher use the motif itself to classify compounds when screening against a dataset. In the following section we explain how spectral masking can help to seperate different structural matches and to refine the motif for further applications such as screening.

## Spectral Masking
Spectral masking applies the masking to a motif. It does it by shifting iteratively every fragment and loss, one after another, to a meaningless mz value in the spectra (just to keep the number of fragments and the given intensity equal to the original spectrum). Everytime when a new mask is applied the similarity between the top database hits and the masked spectrum is calculated. The assumption is that different compounds have a different sensititviy to certain fragments and losses even when the highly intense fragments and losses are present in both.
<img align="center" width="1500" src="masking_anomaly_detection.png">


The changing similarity score over all masked spectra can be compared with all database matches. We found that the Spearman correlation works very well for this comparison. When the Spearman correlation is below a given threshold it can be evident that there are more than one structure type present in the database hits. Using hierachical clustering can be used to seperate them. This can also be an iterative process if the number of different structures is greater than 2.
