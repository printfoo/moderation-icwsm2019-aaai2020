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
    save_path = os.path.join(file_path, "results", "words.csv")
    f = open(save_path, "w")
    f.write("category,words\n")

    lex = pd.read_csv(lex_path)
    lex = lex.dropna(subset = {"name"})
    for name, sdf in lex.groupby("name"):
        ts = "|".join(sdf["tokens"].values)
        if name in ["Swear", "Laugh", "Emoji", "Fake", "Administration", "American", "Nation", "Personal"]:
            f.write(name + "," + ts + "\n")
    f.close()
