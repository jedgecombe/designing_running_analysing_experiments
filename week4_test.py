import logging

import matplotlib.pyplot as plt
import pandas as pd
# import pylab
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('designtime.csv')