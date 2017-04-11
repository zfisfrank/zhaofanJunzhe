# /usr/local/bin/python3

import os
import numpy as np
import pandas as pd

# normalize to 0~1
def normalize(dF):
    dF = (dF - dF.min())/(dF.max() - dF.min())
    dF = dF.dropna(how ='all',axis = 1)
    return dF