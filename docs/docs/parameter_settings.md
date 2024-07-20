# MS2LDA Parameter Settings


## LDA parameter

- depends on gensim or tomotopy implementation; tomotopy easier to understand and interesting parameters (e.g. threshold for word frequency), but gensim has "auto" setting for alpha and eta

## Motif parameter
- how many peaks and losses are included in a motif, should be changed to all
- should the peaks be normalized or not; advantage --> everything is similar; disadvantage --> relative importance gets lost
- normalize over all motifs

## Spec2Vec parameter
- how many molecules should be retrieved
- should the scores be normalized? and then top 10 percent or top 10 be retrieved
- other idea: retrieve best hit -10%; so you can keep the predicted score
- should the same molecule be removed? probably count them but, for later only one --> one could also combine the spectra of the same compounds; 
- include Spec2Vec in negative mode

## Cluster molecules
- use fingerprints for structures to find similar motifs

## Masking
- the current value is set to 1
- we need a spearman threshold to say if two spectra are the same or not
- if more than two than collect disimilar and similar effects based on masking
- how many clusters; this could be done iteratively until the spearman value in one cluster is high enough within the cluster
- how does the finding influence the peak importance for a certain subcluster; how to visualize subclusters
- what difference between mask spectra and the original one is big enough 


## Screening
- The levelA, B,C,D schema needs to be re-evaluated if it is suitable. In the original paper they only look for fragments, we also for losses...