#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np

# get linguistic features
def get_linguistic(p, ts):
    l = len(set(p.split(" ")).intersection(ts))
    return 1 if l > 0 else 0

if __name__ == "__main__":
    
    # get path
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    lex_path = os.path.join(file_path, "resources", "ComLex.csv")
    meta_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta.csv")
    liwc_prep_path = os.path.join(file_path, "data", "youtube_info", "youtube_liwc_prep.csv")
    liwc_path = os.path.join(file_path, "data", "youtube_info", "youtube_liwc.csv")
    save_path = os.path.join(file_path, "data", "youtube_info", "youtube_meta_linguistic.csv")

    # partisan score
    df = pd.read_csv(meta_path)
    df = df.dropna()
    df[["video_id", "time", "token"]].to_csv(liwc_prep_path, index = False)
    lex = pd.read_csv(lex_path)
    lex = lex.dropna(subset = {"name"})
    print(len(lex.groupby("name").count()))

    # calculate linguistic features
    for name, sdf in lex.groupby("name"):
        ts = set("|".join(sdf["tokens"].values).split("|"))
        df["linguist_" + name.lower()] = df["token"].apply(lambda p: get_linguistic(p, ts))
    #df = df.drop(columns = ["token"])
    df.to_csv(save_path, index = False)
    print(df)
