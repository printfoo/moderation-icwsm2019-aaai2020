#!/usr/local/bin/python3

import os, sys, json, random, time
from math import floor, ceil
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
    fig_mod_path = os.path.join(file_path, "results", "separation_sel_mod.pdf")
    fig_bias_path = os.path.join(file_path, "results", "separation_fc_bias.pdf")
    fig_party_path = os.path.join(file_path, "results", "separation_party_thres.pdf")
    fig_degree_path = os.path.join(file_path, "results", "separation_degree_thres.pdf")
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    ate_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate")
    ate_mod_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_self_mod")
    ate_bias_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_factcheck_bias")
    ate_party_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_party_thres")
    ate_degree_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_degree_thres")
    ate_liwc_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_liwc")
    cause_set = ["bias_party"]
    control_set = ["moderated",
                   "meta_like", "meta_dislike", "meta_view",
                   "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
                   "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal",
                   "misinfo_factcheck", "misinfo_veracity", "bias_degree", "bias_party"]

    # plot self moderation
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 0.65))
    for i, cause in enumerate(cause_set):
        plot_x = []
        plot_y = []
        plot_color = []
        for j, per in enumerate(range(6)):
            if per == 0:
                df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
            else:
                df = pd.read_csv(os.path.join(ate_mod_path, cause + "_per" + str(per) + ".csv"))
            ate = df["treated_outcome"] - df["control_outcome"]
            p = 2 * min(len(ate[ate >= 0]), len(ate[ate <= 0])) / len(ate)
            mean = ate.mean()
            lower_ci = ate.quantile(0.025 / 4)
            upper_ci = ate.quantile(1 - 0.025 / 4)
            if p < 0.001:
                if mean > 0:
                    color = "indianred"
                else:
                    color = "cornflowerblue"
            else:
                color = "#555555"
            plot_x.append(mean)
            plot_y.append(per)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax.scatter(per, mean, color = color, linewidth = 0, alpha = 1, s = 30)
            #ax[i].plot([lower_ci, upper_ci], [per, per], color = "k", alpha = 0.9)
        for j, per in enumerate(range(6)):
            ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim([-0.3, 5.3])
        if i > 0:
            ax.set_xticks([])
        ax.set_ylim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax.plot([-0.3, 5.3], [0, 0], color = "k", alpha = 0.4, linestyle = ":")
        if i < 2:
            ax.set_yticks([round(sum(plot_x) / len(plot_x), 2)])
            ax.set_yticklabels(["0%"])
    ax.set_ylim([-0.025, 0.025])
    ax.set_yticks([-0.02, 0, 0.02])
    ax.set_yticklabels(["-2%", "0%", "2%"])
    ax.set_yticklabels(["-2%", "0%", "2%"])
    ax.set_ylabel("H$_0^{sep}$", fontproperties = font2)
    ax.set_xticks([i for i in range(6)])
    ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    ax.set_xlabel("Self-moderation rate", fontproperties = font2)
    plt.savefig(fig_mod_path, bbox_inches = "tight", pad_inches = 0)

    # plot factcheck biass
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 0.65))
    for i, cause in enumerate(cause_set):
        plot_x = []
        plot_y = []
        plot_color = []
        for j, bias in enumerate([-2, -1, 0, 1, 2]):
            if bias == 0:
                df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
            else:
                if cause != "misinfo_veracity":
                    df = pd.read_csv(os.path.join(ate_bias_path, cause + "_bias" + str(bias) + ".csv"))
                else:
                    df = pd.read_csv(os.path.join(ate_bias_path, cause + "_" + str(bias) + "_bias" + str(bias) + ".csv"))
            ate = df["treated_outcome"] - df["control_outcome"]
            p = 2 * min(len(ate[ate >= 0]), len(ate[ate <= 0])) / len(ate)
            mean = ate.mean()
            lower_ci = ate.quantile(0.025 / 4)
            upper_ci = ate.quantile(1 - 0.025 / 4)
            if p < 0.001:
                if mean > 0:
                    color = "indianred"
                else:
                    color = "cornflowerblue"
            else:
                color = "#555555"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax.scatter(bias, mean, color = color, linewidth = 0, alpha = 1, s = 30)
            #ax[i].plot([lower_ci, upper_ci], [bias, bias], color = "k", alpha = 0.9)
        for j, bias in enumerate([-2, -1, 0, 1]):
            if bias < 0:
                ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j])
            else:
                ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j + 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim([-2.2, 2.2])
        if i > 0:
            ax.set_xticks([])
        ax.set_ylim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax.plot([-2.2, 2.2], [0, 0], color = "k", alpha = 0.4, linestyle = ":")
    ax.set_ylim([-0.025, 0.025])
    ax.set_yticks([-0.02, 0, 0.02])
    ax.set_yticklabels(["-2%", "0%", "2%"])
    ax.set_ylabel("H$_0^{sep}$", fontproperties = font2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels([" +2L", "+1L", "0", "+1R", "+2R"])
    ax.set_xlabel("Bias of fact-checkers", fontproperties = font2)
    plt.savefig(fig_bias_path, bbox_inches = "tight", pad_inches = 0)

    # plot degree thresholding
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 0.65))
    for i, cause in enumerate(cause_set):
        plot_x = []
        plot_y = []
        plot_color = []
        for j, bias in enumerate(["0.3", "0.4", "0.5", "0.6", "0.7"]):
            if bias == "0.5":
                df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
            else:
                if cause != "bias_degree":
                    df = pd.read_csv(os.path.join(ate_degree_path, cause + "_degree_thres" + bias + ".csv"))
                else:
                    df = pd.read_csv(os.path.join(ate_degree_path, cause + "_" + bias + "_degree_thres" + bias + ".csv"))
            ate = df["treated_outcome"] - df["control_outcome"]
            p = 2 * min(len(ate[ate >= 0]), len(ate[ate <= 0])) / len(ate)
            mean = ate.mean()
            lower_ci = ate.quantile(0.025 / 4)
            upper_ci = ate.quantile(1 - 0.025 / 4)
            if p < 0.001:
                if mean > 0:
                    color = "indianred"
                else:
                    color = "cornflowerblue"
            else:
                color = "#555555"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax.scatter(bias, mean, color = color, linewidth = 0, alpha = 1, s = 30)
        for j, bias in enumerate([-2, -1, 0, 1]):
            if bias < 0:
                ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j])
            else:
                ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j + 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim([-0.3, 4.3])
        if i > 0:
            ax.set_xticks([])
        ax.set_ylim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax.plot([-0.3, 4.3], [0, 0], color = "k", alpha = 0.4, linestyle = ":")
    ax.set_ylim([-0.025, 0.025])
    ax.set_yticks([-0.02, 0, 0.02])
    ax.set_yticklabels(["-2%", "0%", "2%"])
    ax.set_ylabel("H$_0^{sep}$", fontproperties = font2)
    ax.set_xlabel("Threshold for extreme/center", fontproperties = font2)
    plt.savefig(fig_degree_path, bbox_inches = "tight", pad_inches = 0)

    # plot party thresholding
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 0.65))
    for i, cause in enumerate(cause_set):
        plot_x = []
        plot_y = []
        plot_color = []
        for j, bias in enumerate(["0", "0.05", "0.1", "0.15", "0.2"]):
            if bias == "0":
                df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
            else:
                if cause != "bias_party":
                    df = pd.read_csv(os.path.join(ate_party_path, cause + "_party_thres" + bias + ".csv"))
                else:
                    df = pd.read_csv(os.path.join(ate_party_path, cause + "_" + bias + "_party_thres" + bias + ".csv"))
            ate = df["treated_outcome"] - df["control_outcome"]
            p = 2 * min(len(ate[ate >= 0]), len(ate[ate <= 0])) / len(ate)
            mean = ate.mean()
            lower_ci = ate.quantile(0.025 / 4)
            upper_ci = ate.quantile(1 - 0.025 / 4)
            if p < 0.001:
                if mean > 0:
                    color = "indianred"
                else:
                    color = "cornflowerblue"
            else:
                color = "#555555"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax.scatter(bias, mean, color = color, linewidth = 0, alpha = 1, s = 30)
        for j, bias in enumerate([-2, -1, 0, 1]):
            if bias < 0:
                ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j])
            else:
                ax.plot(plot_y[j: j + 2], plot_x[j: j + 2], color = plot_color[j + 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim([-0.3, 4.3])
        if i > 0:
            ax.set_xticks([])
        ax.set_ylim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax.plot([-0.3, 4.3], [0, 0], color = "k", alpha = 0.4, linestyle = ":")
    ax.set_ylim([-0.025, 0.025])
    ax.set_yticks([-0.02, 0, 0.02])
    ax.set_yticklabels(["-2%", "0%", "2%"])
    ax.set_ylabel("H$_0^{sep}$", fontproperties = font2)
    ax.set_xlabel("Threshold for right/left", fontproperties = font2)
    plt.savefig(fig_party_path, bbox_inches = "tight", pad_inches = 0)
