#!/usr/local/bin/python3

import os, sys, json, random, time, datetime
import pandas as pd

# youtube parser
class youtubeParser:
    
    # initialization
    def __init__(self, raw_path, save_path):
        self.index = 0
        self.raw_path = raw_path
        self.save_path = save_path
        self.f = open(self.save_path, "w")
        self.cols = ["video_id", "channel_id", "channel_title", "view_num",
                    "like_num", "dislike_num", "favorite_num", "comment_num"]
        self.f.write("\t".join(self.cols) + "\n")
    
    # parse records one by one
    def parse(self, res):
        js = json.loads(res)
        save_js = {"video_id": js["videoId"]}
        save_js["channel_id"] = ""
        save_js["channel_title"] = ""
        save_js["view_num"] = ""
        save_js["like_num"] = ""
        save_js["dislike_num"] = ""
        save_js["favorite_num"] = ""
        save_js["comment_num"] = ""
        if len(js["items"]) != 0:
            save_js["video_id"] = js["items"][0]["id"]
            save_js["channel_id"] = js["items"][0]["snippet"]["channelId"]
            try:
                save_js["channel_title"] = js["items"][0]["snippet"]["channelTitle"]
            except:
                pass
            try:
                save_js["view_num"] = js["items"][0]["statistics"]["viewCount"]
            except:
                pass
            try:
                save_js["like_num"] = js["items"][0]["statistics"]["likeCount"]
            except:
                pass
            try:
                save_js["dislike_num"] = js["items"][0]["statistics"]["dislikeCount"]
            except:
                pass
            try:
                save_js["comment_num"] = js["items"][0]["statistics"]["commentCount"]
            except:
                pass
            save_js["favorite_num"] = js["items"][0]["statistics"]["favoriteCount"]
        save_list = [str(save_js[col]) for col in self.cols]
        self.f.write("\t".join(save_list) + "\n")
    
    # traverse all raw responses
    def traverse(self):
        for path_base, _, path_postfixes in os.walk(self.raw_path):
            for path_postfix in path_postfixes:
                with open(os.path.join(path_base, path_postfix), "r") as tmp_f:
                    res = tmp_f.read().strip("\n")
                    self.parse(res)

if __name__ == "__main__":
    
    # get path
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    raw_path = os.path.join(file_path, "data", "youtube_info", "youtube_name_id_raw")
    save_path = os.path.join(file_path, "data", "youtube_info", "youtube_name_id.tsv")

    # parser initilization
    youtube = youtubeParser(raw_path, save_path)
    youtube.traverse()
