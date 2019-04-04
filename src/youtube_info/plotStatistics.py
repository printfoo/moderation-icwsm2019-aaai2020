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
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    comments2_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta_linguistic.csv")
    fig_path = os.path.join(file_path, "results", "youtube_stat_")

    # read
    df = pd.read_csv(comments_path)
    cols = ["bias_party", "bias_degree", "misinfo_veracity", "misinfo_factcheck",
            "meta_like", "meta_dislike", "meta_view",
            "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
            "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal"]
    moderated = df[df["moderated"] == 1]
    living = df[df["moderated"] == 0]
    height = 0.9
    
    # plot
    c_dict = {"bias_party": ["Left", "Right"], "bias_degree": ["Center", "Extreme"],
        "misinfo_veracity": ["False", "True"], "misinfo_factcheck": ["Not", "Fact-checked"],
        "linguist_swear": ["Not", "Swear"], "linguist_laugh": ["Not", "Laugh"],
        "linguist_emoji": ["Not", "Emoji"], "linguist_fake": ["Not", "Fake"],
        "linguist_administration": ["Not", "Admin"], "linguist_american": ["Not", "American"],
        "linguist_nation": ["Not", "Nation"], "linguist_personal": ["Not", "Personal"]}
    for c in c_dict:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(3.4, 0.6))
        bar1 = list(living.sort_values(c).groupby(c).count()["comment"].values)
        bar1 = [b/sum(bar1) for b in bar1]
        bar2 = list(moderated.sort_values(c).groupby(c).count()["comment"].values)
        bar2 = [b/sum(bar2) for b in bar2]
        ax.barh([1, 2], width = [bar1[0], bar2[0]], height = height, color = "k", alpha = 0.3, label = c_dict[c][0])
        ax.barh([1, 2], width = [bar1[1], bar2[1]], height = height, left = [bar1[0], bar2[0]], color = "k", alpha = 0.6, label = c_dict[c][1])
        ax.set_ylim([0.4, 2.6])
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["Unmoderated", "Moderated"], fontproperties = font2)
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("Label distribution", fontproperties = font2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        leg = ax.legend(bbox_to_anchor=(1, 1, 0, 0), loc = "lower right", ncol = 2, borderaxespad = 0.)
        leg.get_frame().set_alpha(0)
        plt.savefig(fig_path + c + ".pdf", bbox_inches = "tight", pad_inches = 0)

    c_dict = {"meta_like", "meta_dislike", "meta_view"}
    for c in c_dict:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(3.4, 0.6))
        bar1 = list(living.sort_values(c).groupby(c).count()["comment"].values)
        bar1 = [b/sum(bar1) for b in bar1]
        bar2 = list(moderated.sort_values(c).groupby(c).count()["comment"].values)
        bar2 = [b/sum(bar2) for b in bar2]
        print(bar1, bar2)
        ax.barh([1, 2], width = [bar1[0], bar2[0]], height = height, color = "k", alpha = 0.2, label = "<$Q_1$")
        ax.barh([1, 2], width = [bar1[1], bar2[1]], height = height, left = [bar1[0], bar2[0]], color = "k", alpha = 0.4, label = "$Q_1$-$Q_2$")
        ax.barh([1, 2], width = [bar1[2], bar2[2]], height = height, left = [bar1[0] + bar1[1], bar2[0] + bar2[1]], color = "k", alpha = 0.6, label = "$Q_2$-$Q_3$")
        ax.barh([1, 2], width = [bar1[3], bar2[3]], height = height, left = [bar1[0] + bar1[1] + bar1[2], bar2[0] + bar2[1] + bar2[2]], color = "k", alpha = 0.8, label = ">$Q_3$")
        ax.set_ylim([0.4, 2.6])
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["Unmoderated", "Moderated"], fontproperties = font2)
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("Label distribution", fontproperties = font2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        leg = ax.legend(bbox_to_anchor=(1, 1, 0, 0), loc = "lower right", ncol = 4, borderaxespad = 0.)
        leg.get_frame().set_alpha(0)
        plt.savefig(fig_path + c + ".pdf", bbox_inches = "tight", pad_inches = 0)
    
    """
    df1 = pd.read_csv(comments_path)
    #print(df1.groupby("channel_id").sum()["moderated"].sort_values())
    for i in [0.25, 0.5, 0.75, 1]:
        print("raw_view", i, df1["raw_view"].quantile(i))
        print("raw_like", i, df1["raw_like"].quantile(i))
        print("raw_dislike", i, df1["raw_dislike"].quantile(i))
    exit()
    for col in ["moderated",
                "misinfo_factcheck", "misinfo_veracity", "bias_party", "bias_degree",
                "meta_like", "meta_dislike", "meta_view",
                "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
                "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal"]:
        mean = df1[col].mean()
        ci = st.t.interval(0.95, len(df1[col])-1, loc=np.mean(df1[col]), scale=st.sem(df1[col]))
        err = (ci[1] - ci[0]) / 2
        print(col, round(mean, 3), "\color{black!50}{\scriptsize $\pm$" + str(round(err, 3)) + "}")
    print()

    df2 = pd.read_csv(comments2_path)
    df2["partisan_bias"] = df2["partisan_degree"] * df2["partisan_party"]
    for col in ["partisan_bias", "partisan_degree", "factcheck_score",
                "video_like_num", "video_dislike_num", "video_view_num"]:
        mean = df2[col].mean()
        ci = st.t.interval(0.95, len(df2[col])-1, loc=np.mean(df2[col]), scale=st.sem(df2[col]))
        err = (ci[1] - ci[0]) / 2
        print(col, mean, err)
    print()
    """

