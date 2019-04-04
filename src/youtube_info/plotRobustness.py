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
    fig_mod_path = os.path.join(file_path, "results", "causality_sel_mod.pdf")
    fig_bias_path = os.path.join(file_path, "results", "causality_fc_bias.pdf")
    fig_party_path = os.path.join(file_path, "results", "causality_party_thres.pdf")
    fig_degree_path = os.path.join(file_path, "results", "causality_degree_thres.pdf")
    fig_liwc_path = os.path.join(file_path, "results", "causality_liwc.pdf")
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    ate_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate")
    ate_mod_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_self_mod")
    ate_bias_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_factcheck_bias")
    ate_party_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_party_thres")
    ate_degree_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_degree_thres")
    ate_liwc_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate_liwc")
    cause_set = ["bias_party", "bias_degree", "misinfo_veracity", "misinfo_factcheck"]
    control_set = ["moderated",
                   "meta_like", "meta_dislike", "meta_view",
                   "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
                   "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal",
                   "misinfo_factcheck", "misinfo_veracity", "bias_degree", "bias_party"]

    # plot self moderation
    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(3.2, 2))
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
                color = "seagreen"
            else:
                color = "indianred"
            plot_x.append(mean)
            plot_y.append(per)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax[i].scatter(mean, per, color = color, linewidth = 0, alpha = 1, s = 30)
            #ax[i].plot([lower_ci, upper_ci], [per, per], color = "k", alpha = 0.9)
        for j, per in enumerate(range(6)):
            ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j])
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].set_ylim([-0.3, 5.3])
        if i > 0:
            ax[i].set_yticks([])
        ax[i].set_xlim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax[i].plot([0, 0], [-0.3, 5.3], color = "k", alpha = 0.4, linestyle = ":")
        if i < 2:
            ax[i].set_xticks([round(sum(plot_x) / len(plot_x), 2)])
            ax[i].set_xticklabels(["0%"])
    ax[0].set_xlim([-0.025, 0.025])
    ax[0].set_xticks([-0.02, 0, 0.02])
    ax[0].set_xticklabels(["-2%", "0%", "2%"])
    ax[1].set_xlim([-0.035, 0.035])
    ax[1].set_xticks([-0.03, 0, 0.03])
    ax[1].set_xticklabels(["-3%", "0%", "3%"])
    ax[2].set_xlim([-0.04, 0.035])
    ax[2].set_xticks([-0.03, 0, 0.03])
    ax[2].set_xticklabels(["-3%", "0%", "3%"])
    ax[3].set_xticks([0, 0.01])
    ax[3].set_xticklabels(["0%", "1%"])
    ax[0].set_xlabel("H1a$_0$", fontproperties = font2)
    ax[1].set_xlabel("H1b$_0$", fontproperties = font2)
    ax[2].set_xlabel("H2a$_0$", fontproperties = font2)
    ax[3].set_xlabel("H2b$_0$", fontproperties = font2)
    ax[0].set_yticks([i for i in range(6)])
    ax[0].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    ax[0].set_ylabel("Self-moderation rate", fontproperties = font2)
    ax[3].scatter(-1, -5, s = 30, color = "indianred", alpha = 1, label = "not rejected")
    ax[3].scatter(-1, -5, s = 30, color = "seagreen", alpha = 1, label = "rejected")
    leg = ax[3].legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_mod_path, bbox_inches = "tight", pad_inches = 0)

    # plot factcheck biass
    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(3.2, 2))
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
            if p < 0.05:
                color = "seagreen"
            else:
                color = "indianred"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax[i].scatter(mean, bias, color = color, linewidth = 0, alpha = 1, s = 30)
            #ax[i].plot([lower_ci, upper_ci], [bias, bias], color = "k", alpha = 0.9)
        for j, bias in enumerate([-2, -1, 0, 1]):
            if bias < 0:
                ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j])
            else:
                ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j + 1])
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].set_ylim([-2.2, 2.2])
        if i > 0:
            ax[i].set_yticks([])
        ax[i].set_xlim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax[i].plot([0, 0], [-2.2, 2.2], color = "k", alpha = 0.4, linestyle = ":")
    ax[0].set_xlim([-0.025, 0.025])
    ax[0].set_xticks([-0.02, 0, 0.02])
    ax[0].set_xticklabels(["-2%", "0%", "2%"])
    ax[1].set_xlim([-0.035, 0.035])
    ax[1].set_xticks([-0.03, 0, 0.03])
    ax[1].set_xticklabels(["-3%", "0%", "3%"])
    ax[2].set_xlim([-0.04, 0.035])
    ax[2].set_xticks([-0.03, 0, 0.03])
    ax[2].set_xticklabels(["-3%", "0%", "3%"])
    ax[3].set_xlim([-0.01, 0.015])
    ax[3].set_xticks([0, 0.01])
    ax[3].set_xticklabels(["0%", "1%"])
    ax[0].set_xlabel("H1a$_0$", fontproperties = font2)
    ax[1].set_xlabel("H1b$_0$", fontproperties = font2)
    ax[2].set_xlabel("H2a$_0$", fontproperties = font2)
    ax[3].set_xlabel("H2b$_0$", fontproperties = font2)
    ax[0].set_yticks([-2, -1, 0, 1, 2])
    ax[0].set_yticklabels([" +2L", "+1L", "0", "+1R", "+2R"])
    ax[0].set_ylabel("Bias of fact-checker", fontproperties = font2)
    ax[3].scatter(-1, -5, s = 30, color = "indianred", alpha = 1, label = "not rejected")
    ax[3].scatter(-1, -5, s = 30, color = "seagreen", alpha = 1, label = "rejected")
    leg = ax[3].legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_bias_path, bbox_inches = "tight", pad_inches = 0)

    # plot degree thresholding
    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(3.2, 2))
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
                color = "seagreen"
            else:
                color = "indianred"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax[i].scatter(mean, bias, color = color, linewidth = 0, alpha = 1, s = 30)
        #ax[i].plot([lower_ci, upper_ci], [bias, bias], color = "k", alpha = 0.9)
        for j, bias in enumerate([-2, -1, 0, 1]):
            if bias < 0:
                ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j])
            else:
                ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j + 1])
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].set_ylim([-0.3, 4.3])
        if i > 0:
            ax[i].set_yticks([])
        ax[i].set_xlim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax[i].plot([0, 0], [-0.3, 4.3], color = "k", alpha = 0.4, linestyle = ":")
    ax[0].set_xlim([-0.025, 0.025])
    ax[0].set_xticks([-0.02, 0, 0.02])
    ax[0].set_xticklabels(["-2%", "0%", "2%"])
    ax[1].set_xlim([-0.035, 0.035])
    ax[1].set_xticks([-0.03, 0, 0.03])
    ax[1].set_xticklabels(["-3%", "0%", "3%"])
    ax[2].set_xlim([-0.04, 0.035])
    ax[2].set_xticks([-0.03, 0, 0.03])
    ax[2].set_xticklabels(["-3%", "0%", "3%"])
    ax[3].set_xlim([-0.01, 0.015])
    ax[3].set_xticks([0, 0.01])
    ax[3].set_xticklabels(["0%", "1%"])
    ax[0].set_xlabel("H1a$_0$", fontproperties = font2)
    ax[1].set_xlabel("H1b$_0$", fontproperties = font2)
    ax[2].set_xlabel("H2a$_0$", fontproperties = font2)
    ax[3].set_xlabel("H2b$_0$", fontproperties = font2)
    #ax[0].set_yticks([-2, -1, 0, 1, 2])
    #ax[0].set_yticklabels([" +2L", "+1L", "0", "+1R", "+2R"])
    ax[0].set_ylabel("Extreme/Center threshold", fontproperties = font2)
    ax[3].scatter(-1, -5, s = 30, color = "indianred", alpha = 1, label = "not rejected")
    ax[3].scatter(-1, -5, s = 30, color = "seagreen", alpha = 1, label = "rejected")
    leg = ax[3].legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_degree_path, bbox_inches = "tight", pad_inches = 0)

    # plot party thresholding
    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(3.2, 2))
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
                color = "seagreen"
            else:
                color = "indianred"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax[i].scatter(mean, bias, color = color, linewidth = 0, alpha = 1, s = 30)
        #ax[i].plot([lower_ci, upper_ci], [bias, bias], color = "k", alpha = 0.9)
        for j, bias in enumerate([-2, -1, 0, 1]):
            if bias < 0:
                ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j])
            else:
                ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j + 1])
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].set_ylim([-0.3, 4.3])
        if i > 0:
            ax[i].set_yticks([])
        ax[i].set_xlim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax[i].plot([0, 0], [-0.3, 4.3], color = "k", alpha = 0.4, linestyle = ":")
    ax[0].set_xlim([-0.025, 0.025])
    ax[0].set_xticks([-0.02, 0, 0.02])
    ax[0].set_xticklabels(["-2%", "0%", "2%"])
    ax[1].set_xlim([-0.035, 0.035])
    ax[1].set_xticks([-0.03, 0, 0.03])
    ax[1].set_xticklabels(["-3%", "0%", "3%"])
    ax[2].set_xlim([-0.04, 0.035])
    ax[2].set_xticks([-0.03, 0, 0.03])
    ax[2].set_xticklabels(["-3%", "0%", "3%"])
    ax[3].set_xlim([-0.01, 0.015])
    ax[3].set_xticks([0, 0.01])
    ax[3].set_xticklabels(["0%", "1%"])
    ax[0].set_xlabel("H1a$_0$", fontproperties = font2)
    ax[1].set_xlabel("H1b$_0$", fontproperties = font2)
    ax[2].set_xlabel("H2a$_0$", fontproperties = font2)
    ax[3].set_xlabel("H2b$_0$", fontproperties = font2)
    #ax[0].set_yticks([-2, -1, 0, 1, 2])
    #ax[0].set_yticklabels([" +2L", "+1L", "0", "+1R", "+2R"])
    ax[0].set_ylabel("Right/Left threshold", fontproperties = font2)
    ax[3].scatter(-1, -5, s = 30, color = "indianred", alpha = 1, label = "not rejected")
    ax[3].scatter(-1, -5, s = 30, color = "seagreen", alpha = 1, label = "rejected")
    leg = ax[3].legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_party_path, bbox_inches = "tight", pad_inches = 0)

    # plot liwc
    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(3.2, 2))
    for i, cause in enumerate(cause_set):
        plot_x = []
        plot_y = []
        plot_color = []
        for j, bias in enumerate(["comlex", "liwc"]):
            if bias == "comlex":
                df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
            else:
                df = pd.read_csv(os.path.join(ate_liwc_path, cause + ".csv"))
            ate = df["treated_outcome"] - df["control_outcome"]
            p = 2 * min(len(ate[ate >= 0]), len(ate[ate <= 0])) / len(ate)
            mean = ate.mean()
            lower_ci = ate.quantile(0.025 / 4)
            upper_ci = ate.quantile(1 - 0.025 / 4)
            if p < 0.01:
                color = "seagreen"
            else:
                color = "indianred"
            plot_x.append(mean)
            plot_y.append(bias)
            plot_color.append(color)
            print("p Value:", round(p, 3))
            ax[i].scatter(mean, bias, color = color, linewidth = 0, alpha = 1, s = 30)
        #ax[i].plot([lower_ci, upper_ci], [bias, bias], color = "k", alpha = 0.9)
        for j, bias in enumerate([0, 1]):
            ax[i].plot(plot_x[j: j + 2], plot_y[j: j + 2], color = plot_color[j])
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].set_ylim([-0.5, 1.5])
        if i > 0:
            ax[i].set_yticks([])
        ax[i].set_xlim([min(plot_x) - 0.01, max(plot_x) + 0.01])
        ax[i].plot([0, 0], [-0.5, 1.5], color = "k", alpha = 0.4, linestyle = ":")
    ax[0].set_xlim([-0.025, 0.025])
    ax[0].set_xticks([-0.02, 0, 0.02])
    ax[0].set_xticklabels(["-2%", "0%", "2%"])
    ax[1].set_xlim([-0.035, 0.035])
    ax[1].set_xticks([-0.03, 0, 0.03])
    ax[1].set_xticklabels(["-3%", "0%", "3%"])
    ax[2].set_xlim([-0.04, 0.035])
    ax[2].set_xticks([-0.03, 0, 0.03])
    ax[2].set_xticklabels(["-3%", "0%", "3%"])
    ax[3].set_xlim([-0.01, 0.015])
    ax[3].set_xticks([0, 0.01])
    ax[3].set_xticklabels(["0%", "1%"])
    ax[0].set_xlabel("H1a$_0$", fontproperties = font2)
    ax[1].set_xlabel("H1b$_0$", fontproperties = font2)
    ax[2].set_xlabel("H2a$_0$", fontproperties = font2)
    ax[3].set_xlabel("H2b$_0$", fontproperties = font2)
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(["ComLex", "LIWC"], rotation = 90, ha = "center", va = "center")
    ax[0].set_ylabel("Lexicon", fontproperties = font2)
    ax[3].scatter(-1, -5, s = 30, color = "indianred", alpha = 1, label = "not rejected")
    ax[3].scatter(-1, -5, s = 30, color = "seagreen", alpha = 1, label = "rejected")
    leg = ax[3].legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_liwc_path, bbox_inches = "tight", pad_inches = 0)
