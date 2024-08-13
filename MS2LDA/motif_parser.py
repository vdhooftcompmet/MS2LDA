import os
from MS2LDA.utils import create_spectrum

def store_m2m_file(motif_spectrum, motif_number, folder):
    """stores one motif spectrum in a .m2m file. It uses the same format as in the original version from MS2LDA.org
    
    ARGS:
        motif_spectrum: matchms spectrum object
        motif_number (int): number that identifies the motif
        folder (str): new folder name with relative or absolute path
    
    RETURNS:
        True
    """

    # Use os.path.basename to get the folder name
    folder_name = os.path.basename(folder).split(" ")[0].lower()
    
    # Construct the filename using os.path.join
    filename = os.path.join(folder, f"{folder_name}_motif_{motif_number}.m2m")

    with open(filename, "w") as output:

        output.write(f"#MS2ACCURACY 0.005\n") # number should be adjustable
        output.write(f"#MOTIFSET {folder.replace(' ', '_')}\n")
        output.write(f"#CHARGE 1\n") # number should be adjustable
        # add name
        output.write(f"#NAME {folder_name}_motif_{motif_number}\n")
        # add (long) annotation
        annotation = motif_spectrum.get("annotation")
        output.write(f"#ANNOTATION {annotation}\n")
        # add (short) annotation
        short_annotation = motif_spectrum.get("short_annotation")
        output.write(f"#SHORT_ANNOTATION {short_annotation}\n")
        # add comment
        comment = motif_spectrum.get("comment")
        output.write(f"#COMMENT {comment}\n")

        for fragment_number in range(len(motif_spectrum.peaks.mz)):
            fragment_mz, fragment_importance = motif_spectrum.peaks.mz[fragment_number], motif_spectrum.peaks.intensities[fragment_number]
            output.write(f"fragment_{fragment_mz},{fragment_importance}\n")

        for loss_number in range(len(motif_spectrum.losses.mz)):
            loss_mz, loss_importance = motif_spectrum.losses.mz[loss_number], motif_spectrum.losses.intensities[loss_number]
            output.write(f"loss_{loss_mz},{loss_importance}\n")

    return True



def store_m2m_folder(motif_spectra, folder):
    """stores a bunch of motif spectra in a new folder where each motif is stored in a .m2m file
    
    ARGS:
        motif_spectra (list): list of matchms spectrum objects
        folder (str): new folder name with relative or absolute path
    
    RETURNS:
        True
    """

    os.makedirs(folder)

    for motif_number, motif_spectrum in enumerate(motif_spectra):
        store_m2m_file(motif_spectrum, motif_number, folder)

    return True



def load_m2m_file(file): # currently it is not supported to change frag/loss tags and significant digits
    """parses mass to motifs by extraction the fragments, losses, names and (short) annotation

    ARGS:
        file (str): path to m2m file

    RETURNS:
        motif_spectrum: matchms spectrum object
    """
    
    features = []
    name = None
    short_annotation = None
    ms2accuracy = None
    motifset = None
    charge = None
    annotation = None

    with open(file, "r") as motif_file:
        for line in motif_file:
            if line.startswith("frag") or line.startswith("loss"):
                features.append( (line.strip().split(",")) )
            elif line.startswith("#NAME"):
                name = line.split("motif_")[1]
            elif line.startswith("#SHORT_ANNOTATION"):
                short_annotation = line.split(" ", 1)[1]
            elif line.startswith("#MS2ACCURACY"):
                ms2accuracy = line.split(" ")[1]
            elif line.startswith("#MOTIFSET"):
                motifset = line.split(" ")[1]
            elif line.startswith("#CHARGE"):
                charge = line.split(" ")[1]
            elif line.startswith("#ANNOTATION"):
                annotation = line.split(" ", 1)[1]

    motif_spectrum = create_spectrum(features, name, frag_tag="fragment_", loss_tag="loss_", significant_digits=2)
    motif_spectrum.set("short_annotation", short_annotation.strip())
    motif_spectrum.set("charge", charge.strip())
    motif_spectrum.set("ms2accuracy", ms2accuracy.strip())
    motif_spectrum.set("motifset", motifset.strip())
    motif_spectrum.set("annotation", annotation.strip())

    return motif_spectrum


def load_m2m_folder(folder):
    """parses an entire folder with m2m files in it
     
    ARGS:
        folder (str): path to m2m folder
        
    RETURNS:
        motif_spectra (list): list of matchms spectrum objects
    """

    files_in_folder = os.listdir(folder)

    motif_spectra = []
    for file in files_in_folder:
        if file.endswith(".m2m"):
            motif_spectrum = load_m2m_file(folder+"\\"+file)
            motif_spectra.append(motif_spectrum)

    return motif_spectra

    

if __name__ == "__main__":
    # load for m2m file
    file = r"..\MotifDB\Euphorbia Plant Mass2Motifs\euphorbia_motif_0.m2m"
    motif_spectrum = load_m2m_file(file)
    print(motif_spectrum)

    # load for m2m folder
    folder = r"..\MotifDB\Euphorbia Plant Mass2Motifs"
    motif_spectra = load_m2m_folder(folder)
    print(motif_spectra[0])

    # store for m2m folder
    folder = r"..\MotifDB\Euphorbia Plant Mass2Motifs Dummy"
    storing = store_m2m_folder(motif_spectra, folder)
    print(storing)
