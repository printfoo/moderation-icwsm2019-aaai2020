#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np
import nltk
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer

# get channel id from url
def get_channel_id(url):
    if "channel/" not in str(url):
        return np.nan
    else:
        return url.split("channel/")[-1]

# tokenize
def tokenize(p):
    tokens = tknzr.tokenize(p) # lower and tokenize
    tokens = [wnl.lemmatize(t, "n") for t in tokens] # lemmatization noun
    tokens = [wnl.lemmatize(t, "v") for t in tokens] # lemmatization verb
    return " ".join(tokens)

if __name__ == "__main__":

    # get path
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    data1_path = os.path.join(file_path, "data", "1st_crawl", "youtube.json")
    data2_path = os.path.join(file_path, "data", "2nd_crawl", "youtube.json")
    factcheck_path = os.path.join(file_path, "data", "2nd_crawl", "factcheck.csv")
    eco_path = os.path.join(file_path, "data", "youtube_info", "misinfo_eco.csv")
    name_path = os.path.join(file_path, "data", "youtube_info", "youtube_name_id.tsv")
    save_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta.csv")

    # partisan score
    eco = pd.read_csv(eco_path)
    eco["channel_id"] = eco["youtube_url"].apply(get_channel_id)
    eco["partisan_score"] = eco["twitter_share_bias"]
    eco["partisan_base"] = eco["twitter_share_base"]
    eco = eco[["channel_id", "partisan_score", "partisan_base"]]
    eco = eco.dropna()
    eco = eco.groupby("channel_id").mean()
    eco["channel_id"] = eco.index
    
    # video to channel and video stat
    name = pd.read_csv(name_path, sep = "\t")
    name = name.dropna(subset = ["channel_id"])
    name["video_view_num"] = name["view_num"]
    name["video_like_num"] = name["like_num"]
    name["video_dislike_num"] = name["dislike_num"]
    name["video_comment_num"] = name["comment_num"]
    name = name[["video_id", "channel_id", "video_view_num", "video_like_num", "video_dislike_num", "video_comment_num"]]
    
    # veracity score
    factcheck = pd.read_csv(factcheck_path)
    factcheck = factcheck[factcheck["site"] == "youtube"]
    factcheck["video_id"] = factcheck["social_id"]
    factcheck["veracity_score"] = factcheck["ruling_val"]
    factcheck = factcheck[["video_id", "veracity_score", "ruling_time", "checkedby"]]
    factcheck = factcheck.dropna()
    factcheck = factcheck.groupby("video_id").min()
    factcheck["video_id"] = factcheck.index
    
    # first crawl
    d1 = pd.read_json(data1_path, lines = True)
    d1 = d1[d1["type"] == "normal"]
    d1["comment_like_num_before"] = d1["likes"]
    d1["video_id"] = d1["id"]
    d1["comment"] = d1["paragraph"]
    d1 = d1[["video_id", "time", "name", "comment", "comment_like_num_before"]]
    d1 = d1.dropna(subset = ["video_id", "time", "comment"])
    d1 = d1.drop_duplicates(subset = ["video_id", "time", "comment"])

    # second crawl
    d2 = pd.read_json(data2_path, lines = True)
    d2 = d2[d2["type"] == "normal"]
    d2["video_id"] = d2["id"]
    d2["comment"] = d2["paragraph"]
    d2 = d2[["video_id", "time", "name", "comment"]]
    d2 = d2.dropna(subset = ["video_id", "time", "comment"])
    d2 = d2.drop_duplicates(subset = ["video_id", "time", "comment"])
    d2["moderated"] = 0 # for left join

    # find df for same video, time, comment
    df = d1.merge(d2, on = ["video_id", "time", "comment"], how = "left")
    df["name_before"] = df["name_x"]
    df["name_now"] = df["name_y"]
    df = df[["video_id", "time", "comment", "moderated", "name_before", "name_now", "comment_like_num_before"]]
    df["moderated"] = df["moderated"].fillna(1)
    
    # merge with other information
    df = df.merge(factcheck, on = "video_id").merge(name, on = "video_id").merge(eco, on = "channel_id")
    
    # tokenize
    wnl = WordNetLemmatizer()
    tknzr = TweetTokenizer(preserve_case = False, reduce_len = True)
    df["token"] = df["comment"].apply(tokenize)

    # cleanup
    df["factcheck_or_not"] = (df["time"] > df["ruling_time"]).apply(lambda x: 1 if x else 0)
    df["factcheck_score"] = df["veracity_score"]
    df["partisan_party"] = df["partisan_score"].apply(lambda x: 1 if x > 0 else -1)
    df["partisan_degree"] = abs(df["partisan_score"])
    df = df[["token", "comment", "time", "moderated", "factcheck_or_not", "factcheck_score", "partisan_party",
             "partisan_degree", "video_like_num", "video_dislike_num", "video_view_num",
             "video_id", "channel_id", "ruling_time", "checkedby"]]
    df = df.dropna()
    df.to_csv(save_path, index = False)
    print(df)

    df = df.drop_duplicates(subset = {"video_id"})
    print(df)
    print(mannwhitneyu(df[df["partisan_party"] == 1]["factcheck_score"], df[df["partisan_party"] == -1]["factcheck_score"]))
    print(df[df["partisan_party"] == 1]["factcheck_score"].mean() - df[df["partisan_party"] == -1]["factcheck_score"].mean())
    print(spearmanr(df["partisan_degree"], df["factcheck_score"]))
