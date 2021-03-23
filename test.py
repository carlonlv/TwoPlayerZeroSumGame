import math
import time
from itertools import chain
import multiprocessing as mp

import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np
import pandas as pd
import tqdm
from sklearn import linear_model

def normalize(x):
    print(x.multi_index)
    return(x)


a = np.arange(11 * 7 * 7).reshape(11, 7, 7)

pool = mp.Pool(processes = 1)
with np.nditer(a , flags = ["multi_index"], op_flags = ["readwrite"]) as it:
    mp_result = pool.map(normalize, it)