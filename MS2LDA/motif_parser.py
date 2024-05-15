import os
from utils import create_spectrum
 

def store_m2m_file(motif_spectrum, motif_number, folder):
    """stores the motif spectra"""
    filename = folder.split("\\")[-1].split(" ")[0].lower()
    with open(f"{folder}\{filename}_motif_{motif_number}.m2m", "w") as output:

        output.write(f"#NAME {filename}_motif_{motif_number}\n")
        output.write(f"ANNOTATION \n")
        output.write(f"SHORT_ANNOTATION \n")
        output.write(f"COMMENT \n")

        for fragment_number in range(len(motif_spectrum.peaks.mz)):
            fragment_mz, fragment_importance = motif_spectrum.peaks.mz[fragment_number], motif_spectrum.peaks.intensities[fragment_number]
            output.write(f"fragment_{fragment_mz},{fragment_importance}\n")

        for loss_number in range(len(motif_spectrum.losses.mz)):
            loss_mz, loss_importance = motif_spectrum.losses.mz[loss_number], motif_spectrum.losses.intensities[loss_number]
            output.write(f"loss_{loss_mz},{loss_importance}\n")

    return True


def store_m2m_folder(motif_spectra, folder):
    """stores motif spectra in a fiven folder"""

    os.makedirs(folder)

    for motif_number, motif_spectrum in enumerate(motif_spectra):
        store_m2m_file(motif_spectrum, motif_number, folder)

    return True


def load_m2m_file(file):
    """parses mass to motifs by extraction the fragments, losses, names and (short) annotation

    ARGS:
        file (str): path to m2m file

    RETURNS:
        motif_spectrum: matchms spectrum object
    """
    
    features = []
    name = None
    short_annotation = None
    #...

    with open(file, "r") as motif_file:
        for line in motif_file:
            if line.startswith("frag") or line.startswith("loss"):
                features.append( (line.strip().split(",")) )
            elif line.startswith("#NAME"):
                name = line.split(" ")[1]
            elif line.startswith("SHORT_ANNOTATION"):
                short_annotation = line.split(" ", 1)[1]
            #...

    motif_spectrum = create_spectrum(features, name, frag_tag="fragment_", loss_tag="loss_")
    motif_spectrum.set("short_annotation", short_annotation)

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
