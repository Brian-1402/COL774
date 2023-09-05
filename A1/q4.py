import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GDA:
    def __init__(self, input_x, input_y):
        mu0 = np.sum(input_x[:, input_y == 0], axis=-1) / np.sum(
            input_y[0, input_y == 0], axis=-1
        )
        mu1 = np.sum(input_x[:, input_y == 1], axis=-1) / np.sum(
            input_y[0, input_y == 1], axis=-1
        )
