#!/usr/local/bin/python3

import os, sys, json, random, time
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
    fig_cor_path = os.path.join(file_path, "results", "correlation_effect.pdf")
    fig_oth_path = os.path.join(file_path, "results", "correlation_other.pdf")
    fig_cau_path = os.path.join(file_path, "results", "causality_effect.pdf")
    fig_res_path = os.path.join(file_path, "results", "causality_results.pdf")
    comments_path = os.path.join(file_path, "data", "youtube_info", "youtube_comments.csv")
    ate_path = os.path.join(file_path, "data", "youtube_info", "youtube_ate")
    cause_set = ["bias_party", "bias_degree", "misinfo_veracity", "misinfo_factcheck"]
    control_set = ["moderated",
                   "meta_like", "meta_dislike", "meta_view",
                   "linguist_swear", "linguist_laugh", "linguist_emoji", "linguist_fake",
                   "linguist_administration", "linguist_american", "linguist_nation", "linguist_personal",
                   "misinfo_factcheck", "misinfo_veracity", "bias_degree", "bias_party"]

    # plot correlation
    df = pd.read_csv(comments_path)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 2))
    w = 0.4
    for i, cause in enumerate(cause_set):
        control = df[df[cause] == 0]["moderated"].mean()
        treated = df[df[cause] == 1]["moderated"].mean()
        control_pc = pc(df[df[cause] == 0]["moderated"].sum(), df[df[cause] == 0]["moderated"].count())
        control_err = control - control_pc[0]
        treated_pc = pc(df[df[cause] == 1]["moderated"].sum(), df[df[cause] == 1]["moderated"].count())
        treated_err = treated - treated_pc[0]
        ax.barh(4 - i - w, width = control, height = w, color = "k", alpha = 0.3)
        ax.plot([control_pc[0], control_pc[1]], [4 - i - w, 4 - i - w], color = "k", alpha = 0.9)
        ax.text(control_pc[1] + 0.001, 4 - i - w, "{0:.2f}".format(round(control * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(control_pc[1] + 0.015, 4 - i - w, r"$\pm${0:.2f}".format(round(control_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        ax.barh(4 - i, width = treated, height = w, color = "k", alpha = 0.6)
        ax.plot([treated_pc[0], treated_pc[1]], [4 - i, 4 - i], color = "k", alpha = 0.9)
        ax.text(treated_pc[1] + 0.001, 4 - i, "{0:.2f}".format(round(treated * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(treated_pc[1] + 0.015, 4 - i, r"$\pm${0:.2f}".format(round(treated_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        print(i, cause, control, treated)
    ax.barh(-1, width = control, height = w, color = "k", alpha = 0.7, label = "group 1")
    ax.barh(-1, width = control, height = w, color = "k", alpha = 0.4, label = "group 0")
    ax.plot([-1, -2], [-5, -6], color = "k", alpha = 0.9, label = "95% CI")
    ax.set_ylim([0.2, 4.3])
    ax.set_yticks([1 - w, 1, 2 - w, 2, 3 - w, 3, 4 - w, 4])
    ax.set_yticklabels(["Not fact-checked", "Fact-checked", "False", "True", "Center", "Extreme", "Left", "Right"], fontproperties = font2)
    ax.set_xlim([0, 0.075])
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_xticklabels(["0%", "1%", "2%", "3%", "4%", "5%"])
    ax.set_xlabel("Moderation likelihood (correlational)", fontproperties = font2, x = 1, ha = "right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    leg = ax.legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_cor_path, bbox_inches = "tight", pad_inches = 0)
    print()

    # plot other correlation
    df = pd.read_csv(comments_path)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7.5, 2))
    lw = 2
    for i, cause in enumerate(cause_set):
        for j, control in enumerate(control_set):
            treat_list = df[df[cause] == 1][control]
            control_list = df[df[cause] == 0][control]
            if "meta_" in control:
                stat, p = mannwhitneyu(treat_list, control_list)
                stat = stat / 1000000000
                print(cause, control, stat, p, treat_list.mean(), control_list.mean())
            else:
                stat, p, dof, n = chi2_contingency([[len(treat_list[treat_list == 1]), len(treat_list[treat_list == 0])],
                                                    [len(control_list[control_list == 1]), len(control_list[control_list == 0])]],
                                                   correction = False)
                print(cause, control, stat, p, treat_list.mean(), control_list.mean())
            if treat_list.mean() > control_list.mean():
                marker = "+"
            else:
                marker = "_"
            if p * 57 < 0.001:
                alpha = 0.9
            elif p * 57 < 0.01:
                alpha = 0.6
            elif p * 57 < 0.05:
                alpha = 0.3
            else:
                alpha = 0.1
            if stat > 0 and stat < 50000 and not (cause == "misinfo_factcheck" and "meta_" in control) and p * 60 < 0.05:
                ax.scatter(j, 4 - i, s = 60, color = "k", marker = marker, linewidth = lw, alpha = alpha)
            elif stat > 0 and stat < 50000 and not (cause == "misinfo_factcheck" and "meta_" in control):
                ax.scatter(j, 4 - i, s = 40, color = "k", linewidth = 0, alpha = alpha)
            else:
                ax.scatter(j, 4 - i, s = 40, color = "k", marker = "x", linewidth = lw, alpha = 0.1)
    ax.plot([0.5, 0.5], [0.5, 5.5], color = "k", alpha = 0.4, linestyle = ":")
    ax.plot([3.5, 3.5], [0.5, 5.5], color = "k", alpha = 0.4, linestyle = ":")
    ax.plot([11.5, 11.5], [0.5, 5.5], color = "k", alpha = 0.4, linestyle = ":")
    ax.text(2, 5.5, "Engagement controls", ha = "center", va = "top", fontproperties = font2)
    ax.text(2, 5, r"(M-W $U$ test)", ha = "center", va = "top", fontproperties = font)
    ax.text(7.5, 5.5, "Linguistic controls", ha = "center", va = "top", fontproperties = font2)
    ax.text(7.5, 5, r"($\chi^2$ test)", ha = "center", va = "top", fontproperties = font)
    ax.text(13.5, 5.5, "Treatments", ha = "center", va = "top", fontproperties = font2)
    ax.text(13.5, 5, r"($\chi^2$ test)", ha = "center", va = "top", fontproperties = font)
    ax.scatter([-1], [-1], s = 60, color = "k", marker = "+", linewidth = lw, alpha = 0.9, label = "positive, $p<0.001$")
    ax.scatter([-1], [-1], s = 60, color = "k", marker = "_", linewidth = lw, alpha = 0.9, label = "negative, $p<0.001$")
    ax.scatter([-1], [-1], s = 60, color = "k", marker = "+", linewidth = lw, alpha = 0.6, label = "positive, $p<0.01$")
    ax.scatter([-1], [-1], s = 60, color = "k", marker = "_", linewidth = lw, alpha = 0.6, label = "negative, $p<0.01$")
    ax.scatter([-1], [-1], s = 60, color = "k", marker = "+", linewidth = lw, alpha = 0.3, label = "positive, $p<0.05$")
    ax.scatter([-1], [-1], s = 60, color = "k", marker = "_", linewidth = lw, alpha = 0.3, label = "negative, $p<0.05$")
    ax.scatter([-1], [-1], s = 40, color = "k", linewidth = 0, alpha = 0.1, label = "not significant")
    ax.scatter([-1], [-1], s = 40, color = "k", marker = "x", linewidth = lw, alpha = 0.1, label = "not applicable")
    ax.set_ylim([0.5, 5.5])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Fact-checked - Not", "True - False", "Extreme - Center", "Right - Left"], fontproperties = font2)
    ax.set_xlim([-0.5, 15.2])
    ax.set_xticks([i for i in range(16)])
    control_set_label = ["Moderation",
               "Like", "Dislike", "View",
               "Swear", "Laugh", "Emoji", "Fake",
               "Administration", "America", "Nation", "Personal",
               "Fact-checked - Not", "True - False", "Extreme - Center", "Right - Left"]
    ax.set_xticklabels(control_set_label, rotation = 30, fontproperties = font2, ha = "right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    leg = ax.legend(bbox_to_anchor=(1, -0.8, 0, 0), loc = "lower right", ncol = 4, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_oth_path, bbox_inches = "tight", pad_inches = 0)
    print()

    # plot causation
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 2))
    for i, cause in enumerate(cause_set):
        this_control = control_set
        df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
        control = df["control_outcome"].mean()
        control_pc = [df["control_outcome"].quantile(0.025), df["control_outcome"].quantile(1 - 0.025)]
        control_err = (control_pc[1] - control_pc[0]) / 2
        treated = df["treated_outcome"].mean()
        treated_pc = [df["treated_outcome"].quantile(0.025), df["treated_outcome"].quantile(1 - 0.025)]
        treated_err = (treated_pc[1] - treated_pc[0]) / 2
        ax.barh(4 - i - w, width = control, height = w, color = "k", alpha = 0.3)
        ax.plot([control_pc[0], control_pc[1]], [4 - i - w, 4 - i - w], color = "k", alpha = 0.9)
        ax.text(control_pc[1] + 0.0008, 4 - i - w, "{0:.2f}".format(round(control * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(control_pc[1] + 0.0125, 4 - i - w, r"$\pm${0:.2f}".format(round(control_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        ax.barh(4 - i, width = treated, height = w, color = "k", alpha = 0.6)
        ax.plot([treated_pc[0], treated_pc[1]], [4 - i, 4 - i], color = "k", alpha = 0.9)
        ax.text(treated_pc[1] + 0.0008, 4 - i, "{0:.2f}".format(round(treated * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 1, fontproperties = font)
        ax.text(treated_pc[1] + 0.0125, 4 - i, r"$\pm${0:.2f}".format(round(treated_err * 100, 2)) + "%",
                ha = "left", va = "center", color = "k", alpha = 0.5, fontproperties = font3)
        print(i, cause, control, treated)
    ax.barh(-1, width = control, height = w, color = "k", alpha = 0.7, label = "treated")
    ax.barh(-1, width = control, height = w, color = "k", alpha = 0.4, label = "controlled")
    ax.plot([-1, -2], [-5, -6], color = "k", alpha = 0.9, label = "95% CI")
    ax.set_ylim([0.2, 4.3])
    ax.set_yticks([1 - w, 1, 2 - w, 2, 3 - w, 3, 4 - w, 4])
    ax.set_yticklabels(["Not fact-checked", "Fact-checked", "False", "True", "Center", "Extreme", "Left", "Right"], fontproperties = font2)
    ax.set_xlim([0, 0.062])
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
    ax.set_xticklabels(["0%", "1%", "2%", "3%", "4%"])
    ax.set_xlabel("Estimated moderation likelihood (causal)", fontproperties = font2, x = 1, ha = "right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    leg = ax.legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_cau_path, bbox_inches = "tight", pad_inches = 0)
    print()

    # plot causality results
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.5, 2))
    for i, cause in enumerate(cause_set):
        df = pd.read_csv(os.path.join(ate_path, cause + ".csv"))
        ate = df["treated_outcome"] - df["control_outcome"]
        p = 2 * min(len(ate[ate >= 0]), len(ate[ate <= 0])) / len(ate)
        mean = ate.mean()
        lower_ci = ate.quantile(0.025 / 4)
        upper_ci = ate.quantile(1 - 0.025 / 4)
        if p < 0.05:
            color = "seagreen"
        else:
            color = "indianred"
        ax.scatter(mean, 4 - i, color = color, linewidth = 0, alpha = 1, s = 60)
        ax.plot([lower_ci, upper_ci], [4 - i, 4 - i], color = "k", alpha = 0.9)
        ax.text(mean + 0.001, 4 - i + 0.15, "{0:.2f}".format(round(mean * 100, 2)) + "%",
                ha = "center", va = "bottom", color = "k", alpha = 1, fontproperties = font)
        ax.text(upper_ci + 0.001, 4 - i, "{0:.2f}".format(round(upper_ci * 100, 2)) + "%]",
                ha = "left", va = "top", color = "k", alpha = 0.5, fontproperties = font3)
        ax.text(lower_ci - 0.0005, 4 - i, "[{0:.2f}".format(round(lower_ci * 100, 2)) + "%",
                ha = "right", va = "top", color = "k", alpha = 0.5, fontproperties = font3)
        print("p Value:", round(p, 3))
    ax.scatter(-1, -5, s = 60, color = "indianred", alpha = 1, label = "not rejected")
    ax.scatter(-1, -5, s = 60, color = "seagreen", alpha = 1, label = "rejected")
    ax.plot([-1, -2], [-5, -6], color = "k", alpha = 0.9, label = "95% CI")
    ax.plot([0, 0], [0.5, 4.5], color = "k", alpha = 0.4, linestyle = ":")
    ax.set_ylim([0.5, 4.5])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Fact-checked - Not", "True - False", "Extreme - Center", "Right - Left"], fontproperties = font2)
    ax.set_xlim([-0.036, 0.025])
    ax.set_xticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02])
    ax.set_xticklabels(["-3%", "-2%", "-1%", "0%", "1%", "2%"])
    ax.set_xlabel("Estimated causal effect (ATE)", fontproperties = font2, x = 1, ha = "right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    leg = ax.legend(bbox_to_anchor=(1, -0.45, 0, 0), loc = "lower right", ncol = 3, borderaxespad = 0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_res_path, bbox_inches = "tight", pad_inches = 0)
