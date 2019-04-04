#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm

# liberal bias
def get_dem(r, d):
    if r["bias_party"] == 0:
        return 1 if r["factcheck_score"] >= 2 - d else 0
    else:
        return 1 if r["factcheck_score"] >= 2 else 0

# conservative bias
def get_rep(r, d):
    if r["bias_party"] == 1:
        return 1 if r["factcheck_score"] >= 2 - d else 0
    else:
        return 1 if r["factcheck_score"] >= 2 else 0

# get party with threshold r
def get_party(x, r):
    if x >= r: return 1
    if x <= -r: return 0
    else: return np.nan

if __name__ == "__main__":
    
    # get path
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta_linguistic.csv")
    liwc_path = os.path.join(file_path, "data", "youtube_info", "youtube_liwc.csv")
    save_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    
    # read
    liwc = pd.read_csv(liwc_path)
    df = pd.read_csv(comments_path)
    linker = ["video_id", "time", "token"]
    sel_col = ["AllPunc", "swear", "money", "work", "bio", "number"]
    liwc = liwc[linker + sel_col]
    for c in sel_col: liwc[c] = liwc[c].apply(lambda x: 1 if x > 0 else 0)
    df = df.merge(liwc, on = linker, how = "inner")
    """
    y = liwc[liwc.columns[3]]
    x = liwc[liwc.columns[4:]]
    x = sm.add_constant(x, prepend = False)
    res = sm.OLS(y, x).fit()
    print(res.summary())
    exit()
    """

    print(df.columns)
    print(len(df.groupby("video_view_num").count()))

    df["moderated"] = df["moderated"].astype("int")
    df["misinfo_factcheck"] = df["factcheck_or_not"].astype("int")
    df["misinfo_veracity"] = df["factcheck_score"].apply(lambda x: 1 if x >= 2 else 0)
    df["bias_party"] = df["partisan_party"].apply(lambda x: 1 if x > 0 else 0)
    df["bias_degree"] = df["partisan_degree"].apply(lambda x: 1 if x > 0.5 else 0)
    df["raw_like"] = df["video_like_num"] / df["video_view_num"]
    df["raw_dislike"] = df["video_dislike_num"] / df["video_view_num"]
    df["raw_view"] = df["video_view_num"]
    for c in ["like", "dislike", "view"]:
        df["meta_" + c] = pd.qcut(df["raw_" + c], 4, labels = False)

    df["bias_party_0.1"] = (df["partisan_party"] * df["partisan_degree"]).apply(lambda x: get_party(x, 0.1))
    print(df.groupby(["bias_party_0.1", "moderated"]).count()["comment"])
    df["bias_party_0.2"] = (df["partisan_party"] * df["partisan_degree"]).apply(lambda x: get_party(x, 0.2))
    print(df.groupby(["bias_party_0.2", "moderated"]).count()["comment"])
    df["bias_party_0.3"] = (df["partisan_party"] * df["partisan_degree"]).apply(lambda x: get_party(x, 0.3))
    print(df.groupby(["bias_party_0.3", "moderated"]).count()["comment"])
    df["bias_party_0.4"] = (df["partisan_party"] * df["partisan_degree"]).apply(lambda x: get_party(x, 0.4))
    print(df.groupby(["bias_party_0.4", "moderated"]).count()["comment"])

    df["bias_degree_0.3"] = df["partisan_degree"].apply(lambda x: 1 if x > 0.3 else 0)
    print(df.groupby(["bias_degree_0.3", "moderated"]).count()["comment"])
    df["bias_degree_0.4"] = df["partisan_degree"].apply(lambda x: 1 if x > 0.4 else 0)
    print(df.groupby(["bias_degree_0.4", "moderated"]).count()["comment"])
    df["bias_degree_0.6"] = df["partisan_degree"].apply(lambda x: 1 if x > 0.6 else 0)
    print(df.groupby(["bias_degree_0.6", "moderated"]).count()["comment"])
    df["bias_degree_0.7"] = df["partisan_degree"].apply(lambda x: 1 if x > 0.7 else 0)
    print(df.groupby(["bias_degree_0.7", "moderated"]).count()["comment"])

    df["misinfo_veracity_-2"] = df.apply(lambda r: get_dem(r, 2), axis = 1)
    df["misinfo_veracity_-1"] = df.apply(lambda r: get_dem(r, 1), axis = 1)
    df["misinfo_veracity_1"] = df.apply(lambda r: get_rep(r, 1), axis = 1)
    df["misinfo_veracity_2"] = df.apply(lambda r: get_rep(r, 2), axis = 1)

    sel_col = ["video_id", "channel_id", "moderated", "comment",
               "misinfo_factcheck", "misinfo_veracity", "bias_party", "bias_degree",
               "meta_like", "meta_dislike", "meta_view", "raw_like", "raw_dislike", "raw_view",
               "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
               "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal",
               "AllPunc", "swear", "money", "work", "bio", "number",
               "misinfo_veracity_-2", "misinfo_veracity_-1", "misinfo_veracity_1", "misinfo_veracity_2",
               "bias_party_0.1", "bias_party_0.2", "bias_party_0.3", "bias_party_0.4",
               "bias_degree_0.3", "bias_degree_0.4", "bias_degree_0.6", "bias_degree_0.7"]
    df = df[sel_col]
    df.to_csv(save_path, index = False)
