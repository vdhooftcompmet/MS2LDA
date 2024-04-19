import pytest
import numpy as np

import sys
sys.path.append('../programming_scripts/')
from SMART_annotation import motifs2tanimotoScore


@pytest.fixture
def motifs():
    motif_A = np.array([1,0,1,0,1,0,1,0,1,0])
    motif_B = np.array([1,1,1,1,1,0,0,0,0,0])
    motif_C = np.zeros(10)
    motif_D = np.array([0,1,0,1,0,1,0,1,0,1])

    return [motif_A, motif_B, motif_C, motif_D]

def test_motifs2tanimotoScore(motifs):
    assert motifs2tanimotoScore(motifs)[0] == 3/7
    assert motifs2tanimotoScore(motifs)[1] == 0/5
    assert motifs2tanimotoScore(motifs)[2] == 0/10
    assert motifs2tanimotoScore(motifs)[3] == 0/5
    assert motifs2tanimotoScore(motifs)[4] == 2/8
    assert motifs2tanimotoScore(motifs)[5] == 0/5
