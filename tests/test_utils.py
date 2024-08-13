import unittest
import sys
import os
import tomotopy as tp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../MS2LDA/')))

##dependancies of the file
from rdkit import Chem
from matchms import Spectrum, Fragments
from matchms.filtering import normalize_intensities
import numpy as np


from matchms import set_matchms_logger_level; set_matchms_logger_level("ERROR")

