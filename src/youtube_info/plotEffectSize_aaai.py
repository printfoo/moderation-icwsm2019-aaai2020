#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_ind, mannwhitneyu, chi2_contingency
from statsmodels.stats.proportion import proportion_confint as pc
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

if __name__ == "__main__":
    
    # get path
    fig_cor_path = os.path.join(file_path, "results", "independence_effect.pdf")
    fig_cau_path = os.path.join(file_path, "results", "separation_effect.pdf")
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    ate_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate")
    cause_set = ["bias_party"]
    control_set = ["moderated",
                   "meta_like", "meta_dislike", "meta_view",
                   "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
                   "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal",
                   "misinfo_factcheck", "misinfo_veracity", "bias_degree", "bias_party"]

    # plot correlation
    df = pd.read_csv(comments_path)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 0.6))
    w = 0.4
    for i, cause in enumerate(cause_set):
        control = df[df[cause] == 0]["moderated"].mean()
        treated = df[df[cause] == 1]["moderated"].mean()
        control_pc = pc(df[df[cause] == 0]["moderated"].sum(), df[df[cause] == 0]["moderated"].count())
        control_err = control - control_pc[0]
        treated_pc = pc(df[df[cause] == 1]["moderated"].sum(), df[df[cause] == 1]["moderated"].count())
        treated_err = treated - treated_pc[0]
        ax.barh(4 - i - w, width = control, height = w, color = "cornflowerblue", alpha = 0.9)
        ax.plot([control_pc[0], control_pc[1]], [4 - i - w, 4 - i - w], color = "k", alpha = 0.5)
        ax.text(control_pc[1] + 0.001, 4 - i - w, "{0:.2f}".format(round(control * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(control_pc[1] + 0.0145, 4 - i - w, r"$\pm${0:.2f}".format(round(control_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        ax.barh(4 - i, width = treated, height = w, color = "indianred", alpha = 0.9)
        ax.plot([treated_pc[0], treated_pc[1]], [4 - i, 4 - i], color = "k", alpha = 0.5)
        ax.text(treated_pc[1] + 0.001, 4 - i, "{0:.2f}".format(round(treated * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(treated_pc[1] + 0.0145, 4 - i, r"$\pm${0:.2f}".format(round(treated_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        print(i, cause, control, treated)
    ax.set_ylim([3.4, 4.2])
    ax.set_yticks([3.6, 4])
    ax.set_yticklabels(["$\hat{\mathbb{P}}\{$M=moderated$\mid$P=left$\}$", "$\hat{\mathbb{P}}\{$M=moderated$\mid$P=right$\}$"], ha = "left", x = -0.95)
    ax.set_xlim([0, 0.075])
    ax.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_xticklabels(["1%", "2%", "3%", "4%", "5%"])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(fig_cor_path, bbox_inches = "tight", pad_inches = 0)
    print()

    # plot causation
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 0.6))
    for i, cause in enumerate(cause_set):
        this_control = control_set
        df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
        control = df["control_outcome"].mean()
        control_pc = [df["control_outcome"].quantile(0.025), df["control_outcome"].quantile(1 - 0.025)]
        control_err = (control_pc[1] - control_pc[0]) / 2
        treated = df["treated_outcome"].mean()
        treated_pc = [df["treated_outcome"].quantile(0.025), df["treated_outcome"].quantile(1 - 0.025)]
        treated_err = (treated_pc[1] - treated_pc[0]) / 2
        ax.barh(4 - i - w, width = control, height = w, color = "cornflowerblue", alpha = 0.9)
        ax.plot([control_pc[0], control_pc[1]], [4 - i - w, 4 - i - w], color = "k", alpha = 0.5)
        ax.text(control_pc[1] + 0.0008, 4 - i - w, "{0:.2f}".format(round(control * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(control_pc[1] + 0.009, 4 - i - w, r"$\pm${0:.2f}".format(round(control_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        ax.barh(4 - i, width = treated, height = w, color = "indianred", alpha = 0.9)
        ax.plot([treated_pc[0], treated_pc[1]], [4 - i, 4 - i], color = "k", alpha = 0.5)
        ax.text(treated_pc[1] + 0.0008, 4 - i, "{0:.2f}".format(round(treated * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(treated_pc[1] + 0.009, 4 - i, r"$\pm${0:.2f}".format(round(treated_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        print(i, cause, control, treated)
    ax.set_ylim([3.4, 4.2])
    ax.set_yticks([3.6, 4])
    ax.set_yticklabels(["$\hat{\mathbb{P}}\{$M=moderated$\mid$P=left, $ps($J$)\}$", "$\hat{\mathbb{P}}\{$M=moderated$\mid$P=right, $ps($J$)\}$"], ha = "left", x = -0.95)
    ax.set_xlim([0, 0.045])
    ax.set_xticks([0.01, 0.02, 0.03])
    ax.set_xticklabels(["1%", "2%", "3%"])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(fig_cau_path, bbox_inches = "tight", pad_inches = 0)
    print()
