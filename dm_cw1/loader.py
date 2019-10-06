'''
# created 06/10/2019 14:22
# by Q.Ducasse
'''

import cv2
import sklearn
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# ============================================
#       CSV FILE LOADING AND VISUALISATION
# ============================================

mtcars = pd.read_csv(path,
                     sep=';',
                     dec=',',
                     na_values='None')
