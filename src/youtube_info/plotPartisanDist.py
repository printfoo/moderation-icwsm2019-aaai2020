#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint as pc
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
sys_path = sys.path[0]
sep = sys_path.find("src")
file_path = sys_path[0:sep]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
font = FontProperties()
font.set_size(8)
font2 = FontProperties()
font2.set_size(8)
font2.set_weight("bold")
font3 = FontProperties()
font3.set_size(7)

# get channel id from url
def get_channel_id(url):
    if "channel/" not in str(url):
        return np.nan
    else:
        return url.split("channel/")[-1]

if __name__ == "__main__":
    
    # get path
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta.csv")
    comments2_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta_linguistic.csv")
    fig_path = os.path.join(file_path, "results", "youtube_bias_dist.pdf")

    # read
    df = pd.read_csv(comments_path)
    df["bias"] = df["partisan_party"] * df["partisan_degree"]
    kde = st.gaussian_kde(df["bias"], bw_method = 0.1)
    
    # plot
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(4, 1.5))
    i = np.linspace(-1,0,100)
    ax.plot(i, kde(i), lw = 2, color = "cornflowerblue")
    i = np.linspace(0,1,100)
    ax.plot(i, kde(i), lw = 2, color = "indianred")
    ax.set_ylim([-0.2, 2.4])
    ax.set_yticks([0, 1, 2])
    #ax.set_yticklabels(["Unmoderated", "Moderated"], fontproperties = font2)
    ax.set_ylabel("Probability density", fontproperties = font2)
    ax.set_xlim([-1.1, 1.1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    #ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Partisanship score", fontproperties = font2)
    ax.plot([0, 0], [-0.2, 2.4], color = "k", alpha = 0.4, linestyle = ":")
    #leg = ax.legend(bbox_to_anchor=(1, 1, 0, 0), loc = "lower right", ncol = 2, borderaxespad = 0.)
    #leg.get_frame().set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(fig_path, bbox_inches = "tight", pad_inches = 0)

