## Annotation of Motifs

**Without annotating a motif, a motif is useless!**
<br>
Even the best motif[^1] that has the potential to identify biochemical relations or important substructures for your research is useless if you don't know that it can explain this relation or substructure. Since it is inevitably that you will get useful, but also useless motifs, it is important to identify those of use.

## How were Motifs annotated in the first MS2LDA implementation?

Manually! Based on fragments and losses in a given motif, expert knowledge was needed to identify which substructure or biochemical relation was represented. Not only the needed expert knowledge was a hinderence for MS2LDA's usabilibity but also its the time consumption needed even for experts in the field.
To simplify the annotation, several tools tried to simplify and help during the annotation. (MAGMA)[https://nlesc.github.io/MAGMa/] was capable of automatically annotating certain features in a motif, but has limitations due to its simplictic algorithm and is rarely capable of annotating a entire motif. (Classyfire)[http://classyfire.wishartlab.com/] helped to identify the chemical class of the motif based on ??. 
Another, very effective way to annotate motifs was to reuse already explored motifs that were stored in a database (MotifDB). During the modeling, it was checked if already known motifs were present in the given dataset. Unfortunately this creates the possibility of getting wrong annotations through selecting the wrong MotifDB dataset. Further you could only select an entire motif set from MotifDB.

## How are Motifs annotated in the new MS2LDA?

Automatically! We use a retrained Spec2Vec model to compare the motif spectra to spectra in a database. We would expect that molecules containing the found molecular substructure have a higher similarity than compounds without it. Sometimes, even good motifs[^1] can represent more than one substructure due to co-occurences of those substructures in the given dataset. When retrieving the top 5 database matches, this can lead to two or more groups of similar molecules. You can read more about it in masking and hierachical clustering.
We could show that on three given datasets we can recall 74 to 82 percent of the manually annotated motifs. This reduces immensively the need for experts to annotate motifs and hopefully pushes the limits in identifying new compounds and their biochemical properties via tandem mass spectrometry!



[^1]: Whatever best or good motif means...
