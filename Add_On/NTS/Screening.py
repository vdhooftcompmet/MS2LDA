import matchms.filtering as msfilters

def spec2mzWords(spectra):
    """converts a spectra into list of frag@xxx loss@xxx spectra

    ARGS:
        spectra (list): list of matchms spectra objects

    RETURNS:
        spectra_set (list): list of lists of strings --> frag@123.45, frag@67.89, ...
    """
    spectra_set = []
    for spectrum in spectra:
        spectrum = msfilters.add_losses(spectrum)
        fragments = spectrum.peaks.mz
        losses = spectrum.losses.mz
        fragments_word = ["frag@"+str(round(mz,2)) for mz in fragments]
        losses_word = ["loss@"+str(round(mz,2)) for mz in losses]
        features_word = fragments_word + losses_word

        spectra_set.append(set(features_word))

    return spectra_set


def screen(motif, spectra_set):
    """finds intersections of query spectra and motif spectra 
    
    ARGS:
        motif: matchms spectrum object
        
    RETURNS:
        overlapping_features (list): list of lists with frag@xxx and loss@ features
        motif_dict (dictionary): dictionary with mz and intensity of motif features
    """

    motif_fragments = ["frag@"+str(mz) for mz in motif.peaks.mz]
    motif_losses = ["loss@"+str(mz) for mz in motif.losses.mz]
    motif_features = motif_fragments + motif_losses
    motif_intensities = list(motif.peaks.intensities) + list(motif.losses.intensities)

    motif_dict = {motif_features[i]: motif_intensities[i] for i in range(len(motif_features))}

    motif = set(motif_features)

    overlapping_features = [motif.intersection(spectrum_set) for spectrum_set in spectra_set]

    return overlapping_features, motif_dict


def screen_score(overlapping_features, motif_dict):
    """based on degree of query spectra-motif spectra overlapp the spectrum gets assigned to a certain level of similarity
    
    ARGS:
        overlapping_features (list): list of lists with frag@xxx and loss@ features
        motif_dict (dictionary): dictionary with mz and intensity of motif features
        
    RETURNS:
        level_A (list): list of indices for 1. best matches
        level_B (list): list of indices for 2. best matches
        level_C (list): list of indices for 3. best matches
        level_D (list): list of indices for 4. best matches
    """

    level_A = []
    level_B = []
    level_C = []
    level_D = []

    
    for idx, features in enumerate(overlapping_features):
        n_features = len(features)
        intensity_sum = 0
        for feature in features:
            match_intensity = motif_dict[feature]
            intensity_sum += match_intensity

        if n_features >= 3 and intensity_sum >= 0.9:
            level_A.append(idx)
        elif n_features == 1 and intensity_sum >= 0.9:
            level_B.append(idx)
        elif n_features >= 2 and intensity_sum >= 0.7:
            level_C.append(idx)
        elif n_features >= 2 and intensity_sum >= 0.5:
            level_D.append(idx)

    return level_A, level_B, level_C, level_D
         
          
    
def run_screen(motif, spectra):
    """runs the screening for a given set of spectra against a motif spectrum
    
    ARGS:
        motif: matchms spectrum object
        spectra (list): list of matchms spectrum objects
        
    RETURNS:
        level_A (list): list of indices for 1. best matches
        level_B (list): list of indices for 2. best matches
        level_C (list): list of indices for 3. best matches
        level_D (list): list of indices for 4. best matches
    """

    spectra_set = spec2mzWords(spectra)
    overlapping_features, motif_dict = screen(motif, spectra_set)
    level_A, level_B, level_C, level_D = screen_score(overlapping_features, motif_dict)

    # things to implement: store as csv file; add retention time; sample run number; compare to mass of analog compound and give suggestions for modifications 
    # maybe there is a more efficient way to do that?
    return level_A, level_B, level_C, level_D

    


if __name__ == "__main__":
    run_screen() # not working