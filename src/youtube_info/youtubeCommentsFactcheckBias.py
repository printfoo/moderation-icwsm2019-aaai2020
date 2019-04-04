#!/usr/local/bin/python3

import os, sys, json, random, time
import pandas as pd
import numpy as np
from causality.estimation.parametric import PropensityScoreMatching

def psm(df, i):
    start_time = time.time()
    cause_set = ["bias_party", "bias_degree", "misinfo_factcheck", "misinfo_veracity_" + str(i)]
    for cause in cause_set:
        confound_dict = {"meta_like": "o", "meta_dislike": "o", "meta_view": "o",
            "linguist_swear": "u", "linguist_laugh": "u", "linguist_emoji": "u", "linguist_fake": "u",
            "linguist_administration": "u", "linguist_american": "u", "linguist_nation": "u", "linguist_personal": "u"}
        for other_cause in cause_set:
            if other_cause != cause:
                confound_dict[other_cause] = "u"
        matcher = PropensityScoreMatching()
        samples = matcher.estimate_ATE(df, cause, "moderated", confound_dict, bootstrap = True)
        samples.to_csv(os.path.join(ate_path, cause + "_bias" + str(i) + ".csv"), index = False)
        print(time.time() - start_time)

if __name__ == "__main__":
    
    # get path
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    ate_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_factcheck_bias")
    df = pd.read_csv(comments_path)
    for i in [-2, -1, 1, 2]:
        psm(df, i)
