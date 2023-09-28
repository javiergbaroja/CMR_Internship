import numpy as np
import pandas as pd
import sys
import os
from natsort import natsorted

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("C:/Users/JAVIER/OneDrive/Escritorio/ETH/Year 2/Spring 2023/Semester Project/CMR_Internship/functions")
from .utils import backslash2slash

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


def set_plot_style(style:str="default"):
    if style in ["latex", "ieee", "nature"]:
        if style == "latex":
            plt.rcParams["font.family"] = "serif"
            plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif']
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
        elif style == "ieee":
            plt.rcParams['axes.prop_cycle']=(cycler('color', ['k', 'r', 'b', 'g']) + cycler('ls', ['-', '--', ':', '-.']))
            plt.rcParams['font.family'] = "serif"
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times', 'Palatino', 'Charter', 'serif']
        else:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams['font.sans-serif']= ['DejaVu Sans', 'Arial', 'Helvetica', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Avant Garde', 'sans-serif']
            plt.rcParams["mathtext.fontset"] = "dejavusans"
    
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["xtick.major.size"] = 3
        plt.rcParams["xtick.major.width"] = 0.5
        plt.rcParams["xtick.minor.size"] = 1.5
        plt.rcParams["xtick.minor.width"] = 0.5
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["ytick.major.size"] = 3
        plt.rcParams["ytick.major.width"] = 0.5
        plt.rcParams["ytick.minor.size"] = 1.5
        plt.rcParams["ytick.minor.width"] = 0.5
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["ytick.right"] = True
    
    elif style == "default":
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams['font.sans-serif']= ['DejaVu Sans', 'Arial', 'Helvetica', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Avant Garde', 'sans-serif']
        plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif']

        plt.rcParams["xtick.direction"] = "out"
        plt.rcParams["xtick.major.size"] = 3.5
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["xtick.minor.size"] = 2.0
        plt.rcParams["xtick.minor.width"] = 0.6
        plt.rcParams["xtick.minor.visible"] = False
        plt.rcParams["xtick.top"] = False
        plt.rcParams["ytick.direction"] = "out"
        plt.rcParams["ytick.major.size"] = 3.5
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["ytick.minor.size"] = 2.0
        plt.rcParams["ytick.minor.width"] = 0.6
        plt.rcParams["ytick.minor.visible"] = False
        plt.rcParams["ytick.right"] = False
    else:
        raise ValueError(f"style must be one of [latex, ieee, nature, default]. Found {style}")


def get_dist_plot_from_csv(path:str, key:str, groupby:str, all:bool=True, plot:str="kde",ymax=1):

    df = pd.read_csv(backslash2slash(path))

    if key not in df.columns or groupby not in df.columns:
        raise ValueError(f"'key' {key} and 'groupby' {groupby} must correspond to one of the 'df' dataframe columns")
    if df[key].dtype != np.int64 and df[key].dtype != float:
        raise TypeError(f"Data in {key} column must be numeric. Input is {df[key].dtype}")
    
    # extract data from whole dataset

    if plot=="kde":
        sns.kdeplot(data=df, x=key, hue=groupby)
        if all:
            sns.kdeplot(data=df, x=key, linewidth=2, color="white")
        plt.vlines(x=df[key].mean(),ymin=0,ymax=ymax,linestyles="--")
    if plot == "violin":
        sns.violinplot(x=groupby, y=key, data=df, jitter=True)


    