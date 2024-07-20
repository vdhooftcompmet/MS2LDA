
## What is a Motif?

In the context of MS2LDA, a motif refers to a recurring pattern or combination of fragments and neutral losses (features) found in mass spectrometry data. These motifs are analogous to topics in text analysis, where certain words frequently co-occur across multiple documents. 
<br>
In tandem mass spectrometry, each spectrum is like a document, and the features are like words. A motif represents a set of features that tend to appear together in many spectra, indicating a common molecular substructure or biochemical relationship.
<br>
By identifying these motifs, scientists can simplify the interpretation of complex mass spectrometry data. Instead of analyzing thousands of individual peaks across numerous spectra, they can focus on a smaller number of motifs. Each motif suggests a recurring biochemical feature, making it easier to understand the underlying chemistry of the sample. 
This approach is particularly useful in studying complex mixtures, where many different compounds are present.

## How are Motifs represented in MS2LDA?

A motif in mass spectrometry consists of fragments and losses, along with their given intensities. This means a motif can be represented as a spectrum, where the importance of features is reflected in the intensity of the features within the spectrum.

To achieve this, we use the [matchms library](https://matchms.readthedocs.io/en/latest/). Matchms is a versatile Python package designed to import, process, clean, and compare mass spectrometry data (MS/MS). <br>
BUT there are differences between a measured spectrum and a motif spectrum:
1. A motif does not have a *precursor mz*
2. A motif has no *retention time*

<br>
One significant advantage of using matchms is its compatibility with many other Python packages for mass spectrometry data, such as [MS2Query](https://github.com/iomega/ms2query), [MassQL](https://mwang87.github.io/MassQueryLanguage_Documentation/), and [Spec2Vec](https://github.com/iomega/spec2vec). 
This compatibility allows for easy integration and data exchange within between the tools (see Annotation and MotifDB) and also potentially for you own workflow!

