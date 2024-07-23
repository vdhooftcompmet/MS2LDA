
import numpy as np
import matplotlib.pyplot as plt



def vis_motifset_similarity(motifset_a, motifset_b, names=["First", "Second"], save=False):
    """compares two set of motifs and draw lines between similar ones
    
    ARGS:
        motifset_a (list): a list of indices which have a similarity to motifset_b
        motifset_b (list): a list of indices which have a similarity to motifset_a
        names (list): names for motifsets
        saves (True or False): parameter if generated plot should be saved
        
    RETURNS:
        None
    """
    
    similarity_sets = list(zip(motifset_a, motifset_b))
    n_motifset_a = list(range(max(motifset_a)+1))
    n_motifset_b = list(range(max(motifset_b)+1))

    unique_entries_a = len(set(motifset_a))
    unique_entries_b = len(set(motifset_b))

    plt.figure(figsize=(25,10))

    plt.text(-0.5, 1.5, f"{names[0]} motifset with {unique_entries_a} unique spectra")
    for i, char in enumerate(n_motifset_a):
        if char in motifset_a:
            plt.text(i, 1.01, char, ha='center',
                    va='center', fontsize=5, color='blue')

    plt.text(-0.5, -1.5, f"{names[1]} motifset with {unique_entries_b} unique spectra")
    for i, char in enumerate(n_motifset_b):
        if char in motifset_b:
            plt.text(i, -1.01, char, ha='center',
                    va='center', fontsize=5, color='red')
    
    for (motifset_a_index, motifset_b_index) in similarity_sets:
        plt.plot([n_motifset_a[motifset_a_index], n_motifset_b[motifset_b_index]], [1,-1], 'k-', lw=0.5)

    plt.xlim(0, max(n_motifset_a + n_motifset_b))
    plt.ylim(-2,2)
    plt.axis('off')

    plt.title("Similarities between to Motif sets")

    if save ==  True:
        plt.savefig(f"{names[0]}_{names[1]}_comparison.jpg")
    plt.show()


if __name__ == "__main__":
    a = [  1,   3,   3,   3,   3,   7,   9,  12,  13,  19,  23,  24,  24,
        26,  26,  26,  28,  32,  32,  33,  39,  43,  43,  43,  43,  43,
        43,  48,  48,  48,  48,  49,  49,  52,  53,  53,  62,  66,  67,
        67,  67,  67,  67,  67,  67,  67,  67,  71,  74,  74,  82,  83,
        83,  83,  83,  83,  83,  83,  83,  83,  83,  83,  83,  84,  84,
        85,  88,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  94,
        99, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102,
       106, 106, 111, 116, 121, 121, 123, 123, 123, 124, 126, 129, 129,
       129, 130, 130, 130, 134, 134, 136, 136, 139, 139, 139, 139, 139,
       139, 139, 139, 139, 139, 139, 140, 140, 140, 140, 141, 141, 148,
       152, 154, 154, 157, 158, 158, 158, 166, 166, 166, 166, 166, 166,
       166, 166, 170, 170, 172, 172, 172, 172, 172, 172, 172, 172, 173,
       173, 173, 173, 173, 173, 173, 180, 180, 180, 180, 180, 187, 188,
       188, 189, 189, 189, 189, 189, 190, 190, 190, 190, 190, 192, 194,
       194, 194, 194, 194, 196, 208, 211, 216, 216, 216, 222, 222, 222,
       222, 223, 226, 227, 230, 233, 234, 240, 241, 241, 243, 243, 243,
       250, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254,
       255, 255, 255, 255, 255, 255, 266, 266, 266, 266, 274, 275, 275,
       275, 275, 276, 276, 278, 285, 285, 285, 288, 293, 293, 293, 294]
    b = [284,  10,  88, 138, 232, 299, 121, 241,  54, 151, 249,  32, 165,
       113, 127, 222, 254,  28, 115,  51, 230,   8,  48, 157, 190, 235,
       290,   4,  11, 164, 181,  33, 167, 290,  65, 208, 174, 118,  22,
        49,  53,  87, 120, 132, 143, 158, 203,   7, 145, 171, 130,  22,
        49,  50,  53,  71,  87, 120, 132, 143, 158, 203, 246,  61, 212,
       295, 287,  30,  89,  96, 106, 117, 119, 173, 176, 229, 279, 284,
        82,  60,  66, 105, 106, 110, 142, 173, 197, 210, 258, 297,  21,
       133, 285, 270, 149,  84, 150, 230, 253, 259, 232, 258,  80, 168,
       262,  57,  83,  94,  77, 115, 107, 234,  13,  36,  64, 158, 166,
       174, 178, 183, 203, 221, 226,  48,  92, 111, 176, 216, 231,  35,
       177,  91, 266, 130, 228, 243, 259,   3,   5,  78, 131, 147, 191,
       242, 273, 108, 270,  58,  92, 182, 199, 250, 260, 262, 268,  19,
        20,  34, 104, 128, 152, 283,  18,  72, 175, 224, 229, 286,  68,
        71,   2,  59, 106, 263, 271,  18, 133, 139, 170, 224, 199,  63,
        85,  99, 100, 286, 266, 281, 246, 201, 206, 240, 126, 165, 265,
       289, 256, 155, 154,  38, 136,  44, 187,  85, 286, 148, 187, 298,
       264,  38,  52,  65,  96, 205, 229, 279, 282, 288, 112, 114, 137,
        79, 141, 261, 264, 287, 294,  74, 227, 254, 276, 256,   1,   9,
        62, 162,  17, 185, 101,   0,  70, 238, 189,  93, 124, 200, 193]

    vis_motifset_similarity(a,b)
