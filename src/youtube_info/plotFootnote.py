#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import statsmodels.api as sm

# get channel id from url
def get_channel_id(url):
    if "channel/" not in str(url):
        return np.nan
    else:
        return url.split("channel/")[-1]

if __name__ == "__main__":
    
    # get path
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    comments2_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta_linguistic.csv")
    data1_path = os.path.join(file_path, "data", "1st_crawl", "youtube.json")
    data2_path = os.path.join(file_path, "data", "2nd_crawl", "youtube.json")
    factcheck_path = os.path.join(file_path, "data", "2nd_crawl", "factcheck.csv")
    eco_path = os.path.join(file_path, "data", "youtube_info", "misinfo_eco.csv")
    name_path = os.path.join(file_path, "data", "youtube_info", "youtube_name_id.tsv")
    
    # read
    df = pd.read_csv(comments_path)
    print(spearmanr(df["meta_like"], df["meta_view"]))
    print(spearmanr(df["meta_dislike"], df["meta_view"]))
    """
    exit()
    m = df.groupby("video_id").sum()[["moderated"]].sort_values("moderated")
    m["video_id"] = m.index
    df = df.merge(m, on = "video_id").sort_values("ruling_time")
    e = df[["video_id", "ruling_time", "moderated_y", "partisan_party"]].drop_duplicates()
    e = e[e["moderated_y"] > 5]
    #print(e)
    print(df[df["video_id"] == "e9geYl9J_Mc"])
    """

    df = pd.read_csv(comments2_path)
    print(df.columns)
    print(spearmanr(df["video_like_num"], df["video_view_num"]))
    print(spearmanr(df["video_dislike_num"], df["video_view_num"]))

    # partisan score
    eco = pd.read_csv(eco_path)
    #eco["channel_id"] = eco["youtube_url"].apply(get_channel_id)
    name = pd.read_csv(name_path, sep = "\t")
    factcheck = pd.read_csv(factcheck_path)
    factcheck["video_id"] = factcheck["social_id"]
    factcheck["veracity_score"] = factcheck["ruling_val"]
    df = factcheck.merge(name, on = "video_id")
    df = df[["video_id", "channel_id", "veracity_score"]].drop_duplicates()
    df = df.merge(eco, on = "channel_id")
    print(df)
    exit()
    df["bias_party"] = df["twitter_share_bias"].apply(lambda x: 1 if x > 0 else 0)
    df = df[["bias_party", "veracity_score"]].dropna()
    print(df)
    print(mannwhitneyu(df[df["bias_party"] == 1]["veracity_score"], df[df["bias_party"] == 0]["veracity_score"]))
    print(df[df["bias_party"] == 1]["veracity_score"].mean() - df[df["bias_party"] == 0]["veracity_score"].mean())

