import numpy as np
import pandas as pd
import sys
import os
from natsort import natsorted

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("C:/Users/JAVIER/OneDrive/Escritorio/ETH/Year 2/Spring 2023/Semester Project/CMR_Internship/functions")
from utils import backslash2slash

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.titlesize'] = BIGGER_SIZE
plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE
plt.rcParams['legend.fontsize'] = SMALL_SIZE
plt.rcParams['figure.titlesize'] = BIGGER_SIZE


def get_dist_plot_from_csv(path:str, key:str, groupby:str, all:bool=True, plot:str="kde"):

    df = pd.read_csv(backslash2slash(path))

    if key not in df.columns or groupby not in df.columns:
        raise ValueError(f"'key' {key} and 'groupby' {groupby} must correspond to one of the 'df' dataframe columns")
    if df[key].dtype != np.int64 and df[key].dtype != float:
        raise TypeError(f"Data in {key} column must be numeric. Input is {df[key].dtype}")
    
    # extract data from whole dataset

    if plot=="kde":
        if all:
            sns.kdeplot(data=df, x=key, hue=groupby)
        sns.kdeplot(data=df, x=key, linewidth=2, color="white")
    if plot == "violin":
        sns.violinplot(x=groupby, y=key, data=df, jitter=True)