"""
UPD 20 June 2026
Maria Kunilovskaya

17 Jun 2026
Using the output of raw_to_data.py (transformed_dataset.tsv),
-- calculate IAA (percentage agreement, Krippendorff alph) for triple annotated items (== sent_ids)
-- apply majority voting and create a gold standard file (sent_id, sent_text, label)

USAGE:
python3 iaa_and_gold.py

"""

import argparse
import os
import sys
import time
from datetime import datetime, date
import krippendorff
from ast import literal_eval
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from collections import Counter


DEFAULTS = {
    "group_mentioned": "no",
    "group_appealed": "NA",
    "multiple_groups": "NA",
    "opposed_groups": "NA",
    "pejorative": "NA",
    "intersectional": "NA",
    "group_tags": ["NA"],
    "reasoning": "NA",
    "polarity": "NA",
    "stance": "NA",
}

# after group_appeal=no
BINARY_FIELDS = [
    "group_mentioned",
    "group_appealed",
    "multiple_groups",
    "opposed_groups",
    "pejorative",
    "intersectional",
]

MULTICLASS_FIELDS = [
    "reasoning",
    "polarity",
    "stance",
]


def adjudicate_id(cur_group):
    # copy data from first annotation
    gold = cur_group.iloc[0].copy()

    gold["annotator"] = "gold"
    gold["updated_at"] = date.today().isoformat()
    gold["adjudicate"] = "no"

    ############################################################
    # 1. group_mentioned
    ############################################################

    value, needs_adj = majority_vote(cur_group["group_mentioned"])

    gold["group_mentioned"] = value

    if needs_adj:
        gold["adjudicate"] = "yes"

    if value != "yes":
        # not applicable: everything else becomes defaults
        for field, default in DEFAULTS.items():
            if field != "group_mentioned":
                gold[field] = default
        return gold
    ############################################################
    # 2. group_appealed
    ############################################################

    value, needs_adj = majority_vote(cur_group["group_appealed"])

    gold["group_appealed"] = value

    if needs_adj:
        gold["adjudicate"] = "yes"

    if value != "yes":
        for field, default in DEFAULTS.items():
            if field not in ("group_mentioned", "group_appealed"):
                gold[field] = default

        return gold
    ############################################################
    # 3. Remaining binary variables
    ############################################################

    for field in (
        "multiple_groups",
        "opposed_groups",
        "pejorative",
        "intersectional",
    ):
        value, needs_adj = majority_vote(cur_group[field])

        gold[field] = value

        if needs_adj:
            gold["adjudicate"] = "yes"

    ############################################################
    # 4. Group tags
    ############################################################

    tags, needs_adj = majority_vote_list(cur_group["group_tags"])

    gold["group_tags"] = tags

    if needs_adj:
        gold["adjudicate"] = "yes"

    ############################################################
    # 5. Remaining multiclass variables
    ############################################################

    for field in (
        "reasoning",
        "polarity",
        "stance",
    ):
        value, needs_adj = majority_vote(cur_group[field])

        gold[field] = value

        if needs_adj:
            gold["adjudicate"] = "yes"

    return gold


def count_disagreement(cur_group, method="relaxed"):
    """
    Count disagreements among three annotations for one sentence.

    Parameters
    ----------
    cur_group : DataFrame
        Three annotations for one sent_id.
    method : {"relaxed", "strict"}

    Returns
    -------
    int
    """

    score = 0

    def disagreement(series):
        vals = pd.unique(series)
        if method == "relaxed":
            return int(len(vals) > 1)
        else:
            return len(vals) - 1

    ##########################################################
    # group_mentioned
    ##########################################################

    score += disagreement(cur_group["group_mentioned"])

    gm, _ = majority_vote(cur_group["group_mentioned"])
    # we don't need to process further
    if gm != "yes":
        return score

    ##########################################################
    # group_appealed
    ##########################################################

    score += disagreement(cur_group["group_appealed"])

    ga, _ = majority_vote(cur_group["group_appealed"])
    if ga != "yes":
        return score

    ##########################################################
    # Remaining binary fields
    ##########################################################

    for field in (
        "multiple_groups",
        "opposed_groups",
        "pejorative",
        "intersectional",
    ):
        score += disagreement(cur_group[field])

    ##########################################################
    # group_tags
    ##########################################################

    if method == "relaxed":

        tag_sets = {
            tuple(sorted(tags))
            for tags in cur_group["group_tags"]
            if isinstance(tags, list)
        }
        # Relaxed:
        # +0 if all annotators assigned exactly the same tag set.
        # +1 if at least one annotator assigned a different tag set.
        score += int(len(tag_sets) > 1)

    else:

        counter = Counter()

        for tags in cur_group["group_tags"]:
            if isinstance(tags, list):
                counter.update(tags)
        # Strict:
        # Count the number of tags that were not assigned
        # by all three annotators (i.e., frequency < 3).
        score += sum(
            count != 3
            for count in counter.values()
        )

    ##########################################################
    # Multiclass fields
    ##########################################################

    for field in (
        "reasoning",
        "polarity",
        "stance",
    ):
        score += disagreement(cur_group[field])

    return score


def percentage_agreement(matrix):
    """
    matrix:
        annotators x items

    Returns:
        agreement_pct,
        n_agree,
        n_usable
    """

    n_agree = 0
    n_usable = 0

    for col in matrix.T:

        vals = col[~pd.isna(col)]

        if len(vals) < 2:
            continue

        n_usable += 1

        if len(set(vals)) == 1:
            n_agree += 1

    agreement_pct = (
        100 * n_agree / n_usable
        if n_usable > 0
        else np.nan
    )

    return agreement_pct, n_agree, n_usable


def tag_coverage_summary(df_scope, interface_df, verbose=False):
    """
    Returns a dataframe with rows:
        - social group type
        - appeal type
        - overall

    and columns:
        seen, unseen, total
    """

    seen_group_tags = set(df_scope["group_tags"].explode().dropna())
    appeal_cols = ["reasoning", "polarity", "stance"]
    seen_appeal_tags = set(df_scope[appeal_cols].stack().dropna())

    group_inventory = set(
        interface_df.loc[
            interface_df["category_order"].between(10, 199),
            "tag",
        ]
    )

    appeal_inventory = set(
        interface_df.loc[
            interface_df["category_order"] >= 200,
            "tag",
        ]
    )

    rows = [
        {
            "category": "social group type",
            "seen": len(seen_group_tags),
            "unseen": len(group_inventory - seen_group_tags),
            "total": len(group_inventory),
        },
        {
            "category": "appeal type",
            "seen": len(seen_appeal_tags),
            "unseen": len(appeal_inventory - seen_appeal_tags),
            "total": len(appeal_inventory),
        },
    ]

    seen_all = seen_group_tags | seen_appeal_tags
    inventory_all = group_inventory | appeal_inventory

    rows.append(
        {
            "category": "overall",
            "seen": len(seen_all),
            "unseen": len(inventory_all - seen_all),
            "total": len(inventory_all),
        }
    )

    if verbose:
        seen_tags = seen_group_tags.union(seen_appeal_tags)

        unseen_group_tags = group_inventory - seen_group_tags
        unseen_appeal_tags = appeal_inventory - seen_appeal_tags
        unseen_tags = unseen_group_tags.union(unseen_appeal_tags)

        unseen_df = (
            interface_df.loc[interface_df["tag"].isin(unseen_tags)]
            .sort_values(["category_order", "group_order"])
        )
        print("Unseen tags:")
        print(
            unseen_df[["category", "group"]]
            .to_string(index=False)
        )

        coverage = (
            interface_df.assign(
                seen=interface_df["tag"].isin(seen_tags)
            )
            .groupby(["category_order", "category"], as_index=False)
            .agg(
                total=("tag", "size"),
                seen=("seen", "sum")
            )
        )

        coverage["unseen"] = coverage["total"] - coverage["seen"]
        coverage["coverage_pct"] = 100 * coverage["seen"] / coverage["total"]

        coverage = coverage.sort_values("category_order").round(1)
        print("\nTag coverage by category:")
        print(
            coverage[["category", "total", "seen", "unseen", "coverage_pct"]]
            .to_string(index=False)
        )

    return pd.DataFrame(rows).set_index("category")


def plot_binary_summary(bin_summary, save_as=None, show=True):
    """
    Plot counts of items with at least one 'yes' annotation.
    """
    plot_df = bin_summary.sort_values("n_items", ascending=False).reset_index(drop=True)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=plot_df,
        x="n_items",
        y="variable",
        ax=ax,
    )
    title = "YES counts for binary annotation decisions"
    ax.set_title(title)
    ax.set_xlabel("Number of items")
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("")

    for i, value in enumerate(plot_df["n_items"]):
        ax.text(
            value + max(plot_df["n_items"]) * 0.01,
            i,
            str(value),
            va="center",
            fontsize=12,
        )

    sns.despine(left=True)

    plt.tight_layout()

    if save_as is not None:
        save_as = Path(save_as)
        save_as.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_as, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


def format_args(args):
    parts = []
    for k, v in vars(args).items():
        if v is None or v is False:
            continue
        if v is True:
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def make_dirs(logs=None, make_them=None, args=None):
    for i in make_them:
        if i:
            Path(i).mkdir(parents=True, exist_ok=True)
    Path(logs).mkdir(parents=True, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = Path(sys.argv[0]).stem
    log_file = Path(logs) / f"{formatted_datetime.split('_')[0]}_{script_name}.log"

    sys.stdout = Logger(logfile=str(log_file))
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    if args:
        print(f"Run settings: python3 {sys.argv[0]} {format_args(args)}")
    else:
        print(f"Run settings: python3 {sys.argv[0]}")


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


PREFIX_TO_CATEGORY = {
    "AGE": "age",
    "GEN": "gender_sexuality",
    "FAM": "family",
    "DIS": "disability",
    "ETH": "ethnicity",
    "MIG": "migration",
    "CIT": "citizenship",
    "REL": "religion",
    "GEO": "geography",
    "EDU": "education",
    "INC": "income",
    "OCC": "occupation",
    "OTH": "non-RISS",
    "REASONING": "reasoning",
    "POLARITY": "polarity",
    "STANCE": "stance"
}


def majority_vote_list(series, threshold=2):
    counter = Counter()

    for tags in series:
        counter.update(tags)

    majority_tags = sorted(
        tag for tag, count in counter.items()
        if count >= threshold
    )

    adjudicate = (len(counter) > 0 and len(majority_tags) == 0)

    return majority_tags, adjudicate


# at least two annotators should return the same label
def majority_vote(series, threshold=2):
    counts = series.value_counts(dropna=False)

    if len(counts) == 0:
        return np.nan, False

    if counts.iloc[0] >= threshold:
        return counts.index[0], False

    return np.nan, True


def diagnose_category(df_iaa=None, prefix='EDU_'):
    tmp = df_iaa.copy()

    tmp["category_value"] = tmp["group_tags"].apply(
        lambda tags: next(
            (tag for tag in tags if tag.startswith(prefix)),
            np.nan
        )
    )

    matrix = tmp.pivot(
        index="annotator",
        columns="sent_id",
        values="category_value"
    )

    for sent_id, col in matrix.items():

        vals = col.dropna()

        if len(vals) < 3:
            print(sent_id, list(vals), "INSUFFICIENT_OVERLAP")

        elif len(set(vals)) == 1:
            print(sent_id, list(vals), "AGREE")

        else:
            print(sent_id, list(vals), "DISAGREE")


# def adjudicate_items(my_df=None, binary=None, multilabel=None, multiclass=None, meth_gold="relaxed",
#                      meth_disagree="relaxed"):
#
#     gold_rows = []
#     for sent_id, group in my_df.groupby("sent_id"):
#         print(sent_id)
#         print(group[["group_mentioned", "group_appealed"]])
#         gold = group.iloc[0].copy()  # use the first annotation as template for gold
#
#         gold["annotator"] = "gold"
#         gold["annotation_id"] = -1
#         gold["updated_at"] = date.today().isoformat()
#
#         gold["adjudicate"] = "no"  # set default value
#         gold["disagreement"] = 0
#         # On how many variables did annotators disagree for current sent_id?
#         n_disagreements = 0
#
#         for variable in binary:
#
#             vals = group[variable].dropna().unique()
#             # yes yes yes -> 0
#             # yes yes no  -> 1
#             if len(vals) > 1:
#                 n_disagreements += 1
#
#             # if adjudicate, gold == None
#             # returns yes/no or np.nan
#             gold[variable], needs_adj_binary = majority_vote(group[variable], threshold=2)
#
#             if needs_adj_binary:
#                 gold["adjudicate"] = "yes"
#
#         for variable in multiclass:
#             if variable == "group_appealed":
#                 vals = group["group_appealed"]
#                 if vals.isna().any():
#                     print(sent_id)
#                     print(group["group_appealed"].map(repr))
#                     exit()
#             # safegarding against annotation errors, avoid looking at items without annotatable content
#             if gold["group_appealed"] == "yes":
#                 vals = pd.unique(group[variable])
#                 if len(vals) > 1:
#                     # A A A -> 0
#                     # A A B -> +1
#                     # A B C -> +2
#                     n_disagreements += len(vals) - 1
#                 try:
#                     gold[variable], needs_adj_binary = majority_vote(group[variable])
#                 except IndexError:
#                     needs_adj_binary = True
#                     gold[variable] = np.nan
#
#                 if needs_adj_binary:
#                     gold["adjudicate"] = "yes"
#         if gold["group_appealed"] == "yes":
#             # multilabel category:
#             # Gold standard is a counterfactual based on majority vote for each tag, i.e.,
#             # if at least 2 annotators assigned a tag, it is included in the gold standard.
#
#             # A: [AGE_CHILD, FAM_FAMILIES]
#             # B: [AGE_CHILD]
#             # C: [FAM_FAMILIES]
#             tags_group, needs_adj_group = majority_vote_list(group[multilabel[0]])
#             gold["group_tags"] = tags_group
#
#             group_tag_sets = {
#                 tuple(sorted(tags))
#                 for tags in group["group_tags"]
#                 if isinstance(tags, list)
#             }
#             if needs_adj_group:
#                 gold["adjudicate"] = "yes"
#             if meth_disagree == 'relaxed':
#                 # On how many variables did annotators disagree?
#                 # [A]
#                 # [A]
#                 # [B] = +1
#
#                 # [A, B, C]
#                 # [A]
#                 # [D] = +1
#                 if len(group_tag_sets) > 1:
#                     n_disagreements += 1
#             else:
#                 # how many labels do not have majority agreement?
#                 # [A][A][B] = +1, adj = no
#                 # [A,B,C][A][D] = +3, adj = no
#                 counter = Counter()
#
#                 for tags in group["group_tags"]:
#                     counter.update(tags)
#
#                 n_disagreements += sum(
#                     1
#                     for count in counter.values()
#                     if count != 3
#                 )
#
#         gold["disagreement"] = n_disagreements
#         gold_rows.append(gold)
#
#     _gold_df = pd.DataFrame(gold_rows)
#
#     return _gold_df


def iaa_binary_multiclass(my_df, my_vars):
    df_iaa = my_df.copy()

    df_iaa["updated_at"] = pd.to_datetime(df_iaa["updated_at"])
    df_iaa = (
        df_iaa.sort_values("updated_at")
        .groupby(["sent_id", "annotator"], as_index=False, observed=True)  # only uses combinations that actually occur
        .tail(1)
    )
    # ----------------------------
    # Caregory-level IAA for binary and multiclass variables
    # ----------------------------
    results = []

    for variable in my_vars:
        n_values = df_iaa[variable].nunique(dropna=True)
        n_items = df_iaa["sent_id"].nunique()
        matrix = (
            df_iaa
            .pivot(
                index="annotator",
                columns="sent_id",
                values=variable
            )
        )

        unique_values = pd.unique(matrix.values.ravel())
        unique_values = [v for v in unique_values if pd.notna(v)]

        alpha = np.nan
        agreement_pct = np.nan

        if len(unique_values) <= 1:
            status = "single_value"

        else:

            value_map = {
                value: i
                for i, value in enumerate(sorted(unique_values))
            }

            matrix_num = matrix.apply(
                lambda col: col.map(value_map)
            )

            alpha = krippendorff.alpha(
                reliability_data=matrix_num.values,
                level_of_measurement="nominal"
            )

            agreement_pct, n_agree, n_usable = percentage_agreement(matrix.values)

            status = "undefined" if np.isnan(alpha) else "ok"

        results.append({
            "variable_type": "binary" if n_values == 2 else "multiclass",
            "variable": variable,
            "agree_pct": round(agreement_pct, 2),
            "alpha": round(alpha, 3) if pd.notna(alpha) else np.nan,
            "n_items": n_items,
            "n_values": n_values,
            # "status": status,
        })

    iaa_df = pd.DataFrame(results)

    return iaa_df


def iaa_multilabel_tags(my_df, multilabel, ordering_df, min_freq=2):
    df_iaa = my_df.copy()
    # ----------------------------
    # multilabel variables (tag-wise, treating each tag as binary choice)
    # How often do annotators agree on the presence of each group tag, regardless of other tags?
    # ----------------------------

    n_items = df_iaa["sent_id"].nunique()
    tag_results = []
    all_tags = sorted(
        set(
            tag
            for tags in df_iaa[multilabel].dropna()
            for tag in tags
        )
    )

    for tag in all_tags:
        # exclude infrequent tag in the triple-annotated dataset (e.g., only one occurrence)
        positive_count = sum(
            tag in tags
            for tags in df_iaa[multilabel]
        )

        if positive_count < min_freq:
            tag_results.append({
                "variable_type": "multilabel",
                "variable": tag,
                "agree_pct": np.nan,
                "alpha": np.nan,
                "tag_freq": int(positive_count),  # how many sentences have this tag
                "n_items": n_items,
                "n_values": 2,
                "status": "too_rare"
            })
            continue
        tmp = df_iaa.copy()

        tmp[f"tag_{tag}"] = tmp[multilabel].apply(
            lambda tags:
            "yes"
            if isinstance(tags, list) and tag in tags
            else "no"
        )

        matrix = (
            tmp
            .pivot(
                index="annotator",
                columns="sent_id",
                values=f"tag_{tag}"
            )
            .apply(lambda col: col.map({"yes": 1, "no": 0}))
        )

        unique_values = pd.unique(matrix.values.ravel())
        unique_values = [v for v in unique_values if pd.notna(v)]

        alpha = np.nan
        if len(unique_values) <= 1:
            status = "single_value"
        else:
            alpha = krippendorff.alpha(
                reliability_data=matrix.values,
                level_of_measurement="nominal"
            )
            status = "ok"

        agreement_pct, n_agree, n_usable = percentage_agreement(matrix.values)

        tag_results.append({
            "variable_type": "multilabel",
            "variable": tag,
            "agree_pct": agreement_pct,
            "alpha": alpha,
            "n_items": n_items,
            "n_values": 2,
            "tag_freq": positive_count,
            "status": status
        })

    tag_iaa_df = pd.DataFrame(tag_results)

    tag_iaa_df = tag_iaa_df.merge(
        ordering_df,
        left_on="variable",
        right_on="tag",
        how="left",
        indicator=True,
    )

    tag_iaa_df["variable"] = tag_iaa_df["group"]

    tag_iaa_df["variable"] = tag_iaa_df["group"]  # this group comes from interface choices
    # Which tags did not match the ordering table?

    tag_iaa_df = (
        tag_iaa_df
        .sort_values(["category_order", "group_order"])
        .drop(columns=["tag"])
        .reset_index(drop=True)
    )

    tag_iaa_df = tag_iaa_df[["variable_type", "category", "variable",
                             "agree_pct", "alpha", "tag_freq",
                             "n_items", "n_values", "status"]]
    tag_iaa_df["tag_freq"] = tag_iaa_df["tag_freq"].astype("Int64")  # keeps missing values as <NA> not NaN

    return tag_iaa_df


def iaa_multilabel_categories(my_df, multilabel, cat_pref=None, meth="library"):
    df_iaa = my_df.copy()
    # debugging duplicate annotators: in team
    # dups = df_iaa[df_iaa.duplicated(["annotator", "sent_id"], keep=False)]
    # print(dups[["sent_id", "annotator"]])
    # print(dups.sort_values(["sent_id", "annotator"]))
    # exit()
    # ----------------------------
    # Multilabel variable (category-wise)
    # ----------------------------
    cat_results = []

    # diagnose_category(df_iaa=df_iaa, prefix='ETH_')
    # exit()

    for prefix in cat_pref:

        tmp = df_iaa.copy()

        tmp["category_present"] = tmp[multilabel].apply(
            lambda tags: "yes" if any(tag.startswith(prefix) for tag in tags) else "no"
        )

        matrix = tmp.pivot(
            index="annotator",
            columns="sent_id",
            values="category_present",
        )

        matrix = matrix.eq("yes").astype(int)

        unique_values = pd.unique(matrix.values.ravel())
        unique_values = [v for v in unique_values if pd.notna(v)]

        n_usable_items = (matrix.notna().sum(axis=0).ge(2).sum())

        usable_cols = matrix.columns[matrix.notna().sum(axis=0) >= 2]

        usable_matrix = matrix[usable_cols]

        value_counts = (pd.Series(usable_matrix.values.ravel()).dropna().value_counts())
        usable_values = value_counts[value_counts >= 2]

        alpha = np.nan
        agreement_pct = np.nan
        if n_usable_items == 0:
            # There are not enough items for which at least two annotators provided a value for that category.
            status = "insufficient_overlap"
        elif len(value_counts) <= 1:
            status = "single_value"
        elif len(usable_values) < 2:
            # Category-level α was not computed when fewer than two category values occurred at least twice.
            status = "too_sparse"
        else:
            if meth == 'library':
                value_map = {value: i for i, value in enumerate(sorted(unique_values))}

                matrix_num = matrix.apply(lambda col: col.map(value_map))
                alpha = krippendorff.alpha(reliability_data=matrix_num.values, level_of_measurement="nominal")
            elif meth == 'manual':
                exit('Manual calculation of Krippendorff\'s alpha is not implemented yet.')
            else:
                raise ValueError("Invalid method for Krippendorff's alpha calculation.")
            status = "ok"

            agreement_pct, n_agree, n_usable = percentage_agreement(matrix.values)

        cat_results.append({
            "variable_type": "multilabel",
            "category": PREFIX_TO_CATEGORY[prefix[:-1]],
            "agree_pct": agreement_pct,
            "alpha": alpha,
            "n_items": n_usable_items,
            "n_values": len(value_counts),
            # "status": status,
        })
    cat_iaa_df = pd.DataFrame(cat_results)

    return cat_iaa_df


def one_row_summary(iaa_res_bin, iaa_res_multi, cat_iaa_res, cross_annotated):
    collector = [{
        # ("", "# triple"): cross_annotated["sent_id"].nunique(),
        ("Binary", "Agree %"): iaa_res_bin["agree_pct"].mean(),
        ("Binary", "Alpha"): iaa_res_bin["alpha"].mean(),
        # ("", "# triple"): cross_annotated2["sent_id"].nunique(),

        ("Multiclass", "Agree %"): iaa_res_multi["agree_pct"].mean(),
        ("Multiclass", "Alpha"): iaa_res_multi["alpha"].mean(),

        ("Multilabel", "Agree %"): cat_iaa_res["agree_pct"].mean(),
        ("Multilabel", "Alpha"): cat_iaa_res["alpha"].mean(),
        ("Overall", "Agree %"): pd.concat([
            iaa_res_bin["agree_pct"],
            iaa_res_multi["agree_pct"],
            cat_iaa_res["agree_pct"],
        ]).mean(),
        ("Overall", "Alpha"): pd.concat([
            iaa_res_bin["alpha"],
            iaa_res_multi["alpha"],
            cat_iaa_res["alpha"],
        ]).mean(),
    }]

    summary_df = (
        pd.DataFrame.from_records(collector)
        .round({
            ("Binary", "Agree %"): 2,
            ("Binary", "Alpha"): 3,
            ("Multilabel", "Agree %"): 2,
            ("Multilabel", "Alpha"): 3,
            ("Multiclass", "Agree %"): 2,
            ("Multiclass", "Alpha"): 3,
            ("Overall", "Agree %"): 2,
            ("Overall", "Alpha"): 3,
        })
    )

    return summary_df


def detailed_summary(iaa_res_bin, iaa_res_multi, cat_iaa_res):
    rows = []
    # Binary
    rows.append(iaa_res_bin)
    rows.append(pd.DataFrame([{
        "variable_type": "Average",
        "variable": "",
        "agree_pct": iaa_res_bin["agree_pct"].mean(),
        "alpha": iaa_res_bin["alpha"].mean(),
        "n_items": "--",
        "n_values": "--",
    }]))

    # Multiclass
    rows.append(iaa_res_multi)
    rows.append(pd.DataFrame([{
        "variable_type": "Average",
        "variable": "",
        "agree_pct": iaa_res_multi["agree_pct"].mean(),
        "alpha": iaa_res_multi["alpha"].mean(),
        "n_items": "--",
        "n_values": "--",
    }]))

    # Multilabel
    rows.append(cat_iaa_res.rename(columns={"category": "variable"}))
    rows.append(pd.DataFrame([{
        "variable_type": "Average",
        "variable": "",
        "agree_pct": cat_iaa_res["agree_pct"].mean(),
        "alpha": cat_iaa_res["alpha"].mean(),
        "n_items": "--",
        "n_values": "--",
    }]))

    summary_df = pd.concat(rows, ignore_index=True)

    summary_df = pd.concat([
        summary_df,
        pd.DataFrame([{
            "variable_type": "Overall",
            "variable": "",
            "agree_pct": pd.concat([
                iaa_res_bin["agree_pct"],
                iaa_res_multi["agree_pct"],
                cat_iaa_res["agree_pct"],
            ]).mean(),
            "alpha": pd.concat([
                iaa_res_bin["alpha"],
                iaa_res_multi["alpha"],
                cat_iaa_res["alpha"],
            ]).mean(),
            "n_items": "--",
            "n_values": "--",
        }])
    ], ignore_index=True)

    summary_df = summary_df.round({"agree_pct": 2, "alpha": 3})

    return summary_df


RUN = "main_student_groups"  # "main_student_groups", "trial_student_groups"
my_date = "30June2026"  # 29June2026, 16June2026
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # these should be portable
    parser.add_argument('--intab', help="", default=f'res/{RUN}/{my_date}_transformed_dataset.tsv')
    parser.add_argument('--interface', help="", default=f'interface/{RUN}/interface_scheme.tsv')
    parser.add_argument('--gold_to', default=f'data/{RUN}/')
    parser.add_argument('--res', default=f'res/{RUN}/iaa/')
    parser.add_argument('--reworks', default=f'res/{RUN}/reworks/')
    parser.add_argument('--pics', default=f'pics/{RUN}/')
    parser.add_argument('--logs', default=f'logs/{RUN}/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics, args.gold_to, args.reworks], args=args)

    df = pd.read_csv(args.intab, sep='\t')
    # pandas might read our "NA" as NaN
    df["group_tags"] = df["group_tags"].apply(lambda x: literal_eval(x) if x else ["NA"])
    df = df.fillna("NA")

    print(df[["annotator", "sent_id", "group_mentioned", "group_appealed", "group_tags", "reasoning", "stance"]].head())

    to_rework = df.loc[df["incomplete"] == "yes", ["annotator", "sent_id", "left_context", "text", "right_context",
                                                   "further_context", "date", "dataset", "year"]]
    to_rework.to_csv(os.path.join(args.reworks, "faulty_incomplete_annotations.tsv"), index=False, sep='\t')

    # drop them for the IAA analysis and gold generation
    df = df[df["incomplete"] != "yes"]

    # export sent_ids for multiple groups tagged:
    multi_group = (
        df[df["multiple_groups"] == "yes"]
        [["sent_id", "left_context", "text", "right_context",
          "further_context", "date", "dataset", "year"]]
        .drop_duplicates("sent_id")
    )
    print("How many annotations indicated multiple groups?")
    print(df["multiple_groups"].value_counts(dropna=False))
    print('How many sent_ids have multiple groups tagged at least once?')
    print(len(multi_group))
    multi_group.to_csv(os.path.join(args.reworks, "multi_group_sent_ids.tsv"), index=False)

    # ---- Subsetting data for IAA and adjudication ----
    triple_or_more_annotated_items = df.groupby("sent_id").filter(lambda x: x["annotation_id"].nunique() >= 3)

    print(f"\nTotal annotations: {len(df)}")
    print(f"Sentences with 3+ annotations: {len(triple_or_more_annotated_items['sent_id'].unique())}")
    print(f"\t--- after removing {len(to_rework)} incomplete/faulty annotations ---")
    # from each set of annotations take random three
    cross_annotated = (
        triple_or_more_annotated_items
        .groupby("sent_id", group_keys=False)
        .sample(n=3, random_state=42)
        .reset_index(drop=True)
    )

    # # --- INSPECT UNSEEN TAGS ---
    interface_df = pd.read_csv(args.interface, sep='\t')
    interface_df = interface_df[~interface_df["tag"].str.contains("_*", regex=False, na=False)]
    triple_coverage = tag_coverage_summary(cross_annotated, interface_df, verbose=False)

    print("\nTag coverage in triple-annotated data")
    print(triple_coverage)
    triple_coverage.to_csv(os.path.join(args.res, "triple_tag_coverage.csv"), sep='\t', index=False)

    # define types of variables for IAA and gold standard generation
    class_vars = ['intersectional', 'multiple_groups', 'opposed_groups', 'pejorative',
                  'reasoning', 'polarity', 'stance']

    # ======= IAA ========
    # calculate IAA true global binary: group_mentioned
    iaa_res_mention = iaa_binary_multiclass(cross_annotated, ["group_mentioned"])

    # calculate IAA where group_mentioned == "yes" in all annotations
    cross_annotated1 = cross_annotated.groupby("sent_id").filter(lambda x: (x["group_mentioned"] == "yes").all())
    iaa_res_appealed = iaa_binary_multiclass(cross_annotated1, ["group_appealed"])

    # keep sent_ids where group_appealed == "yes" in all annotations
    cross_annotated2 = cross_annotated1.groupby("sent_id").filter(lambda x: (x["group_appealed"] == "yes").all())
    iaa_res_multi = iaa_binary_multiclass(cross_annotated2, class_vars)

    if RUN == "trial_student_groups":
        cat_prefixes = ["AGE_", "GEN_", "FAM_", "DIS_", "ETH_", "MIG_", "REL_", "GEO_", "EDU_", "INC_", "OCC_", "OTH_"]
    else:
        cat_prefixes = ["AGE_", "GEN_", "FAM_", "DIS_", "ETH_", "MIG_", "CIT_",
                        "REL_", "GEO_", "EDU_", "INC_", "OCC_", "OTH_"]

    cat_iaa_res = iaa_multilabel_categories(cross_annotated2, "group_tags", cat_pref=cat_prefixes, meth="library")

    iaa_res_bin = pd.concat([iaa_res_mention, iaa_res_appealed], ignore_index=True)
    mini_res = one_row_summary(iaa_res_bin, iaa_res_multi, cat_iaa_res, cross_annotated)
    print()
    print(mini_res.to_string(index=False))

    print("\nDetailed IAA results:")
    detailed = detailed_summary(iaa_res_bin, iaa_res_multi, cat_iaa_res)
    print(detailed)
    detailed.to_csv(os.path.join(args.res, "iaa_summary.tsv"), sep='\t', index=False)
    print(f"IAA summary saved to {args.res}iaa_summary.tsv")

    # separately look at IAA per-tag in group-tags
    tag_iaa_res = iaa_multilabel_tags(cross_annotated2, "group_tags", interface_df, min_freq=5)

    # for per-tag view: keep only rows with valid alpha
    tag_iaa_res_valid = tag_iaa_res.dropna(subset=["alpha"]).reset_index(drop=True)
    # count rows with missing alpha
    n_dropped = tag_iaa_res["alpha"].isna().sum()

    print(f"\nPer-tag IAA is not defined for {n_dropped} (of {len(tag_iaa_res)})")
    tag_iaa_res_valid = tag_iaa_res_valid.drop(["status"], axis=1)
    # aggregate: add Avg row and round as before
    tag_freq_mean = pd.to_numeric(
        tag_iaa_res_valid["tag_freq"], errors="coerce"
    ).mean()
    avg_row = pd.DataFrame({
        "variable_type": ["Average"],
        "category": [""],
        "variable": [""],
        "agree_pct": [round(tag_iaa_res_valid["agree_pct"].mean(), 2)],
        "alpha": [round(tag_iaa_res_valid["alpha"].mean(), 3)],
        "tag_freq": [round(tag_freq_mean, 1) if pd.notna(tag_freq_mean) else pd.NA],
        # "tag_freq": [round(tag_iaa_res_valid["tag_freq"].mean(), 1)],
        # "agree_pct": [tag_iaa_res_valid["agree_pct"].mean().round(2)],
        # "alpha": [tag_iaa_res_valid["alpha"].mean().round(3)],
        # "tag_freq": [tag_iaa_res_valid["tag_freq"].mean().round(1)],
        "n_items": ["--"],
        "n_values": ["--"],
    })

    tag_valid_out = pd.concat([tag_iaa_res_valid, avg_row], ignore_index=True)
    # tag_valid_out = tag_valid_out.astype({"n_items": "Int64", "n_values": "Int64"})
    print(f"\nPer-tag IAA results (valid alpha) for {len(tag_iaa_res_valid)} tags:")
    print(tag_valid_out.to_string(index=False))

    tag_valid_out.to_csv(os.path.join(args.res, "iaa_results_tag-level_social-groups_valid.tsv"),
                         sep='\t', index=False)

    tag_iaa_res["alpha"] = tag_iaa_res["alpha"].round(3)
    tag_iaa_res["agree_pct"] = tag_iaa_res["agree_pct"].round(2)
    tag_iaa_res.to_csv(os.path.join(args.res, "iaa_results_tag-level_social-groups_all.tsv"), sep='\t', index=False)
    print(f"IAA full results per tag + status are saved separately.\n")

    # --- RARE BUT SEEN TAGS: < 5 freq tags are skipped for IAA (alpha = NaN) ---
    low_freq_report = (
        tag_iaa_res.loc[tag_iaa_res["variable_type"] == "multilabel"]
        .assign(low_freq=lambda x: x["alpha"].isna())
        .groupby("category")
        .agg(
            low_freq=("low_freq", "sum"),
            total=("variable", "size"),
        )
        .reset_index()
    )

    low_freq_report["low_freq/total"] = (
            low_freq_report["low_freq"].astype(str)
            + "/"
            + low_freq_report["total"].astype(str)
    )

    low_freq_report["pct"] = (
            100 * low_freq_report["low_freq"] / low_freq_report["total"]
    ).round(1)

    low_freq_report = (
        low_freq_report[
            ["category", "low_freq/total", "pct"]
        ]
        .sort_values("pct", ascending=False)
    )
    cat_interface_df = interface_df[["category", "category_order"]].drop_duplicates()
    low_freq_report = (
        low_freq_report
        .merge(cat_interface_df, on="category", how="left")
        .sort_values("category_order")
        .drop(columns="category_order")
    )
    print("\nReport on low-frequency tags (skipped for IAA calculation):")
    print(low_freq_report.to_string(index=False))

    low_freq_report.to_csv(os.path.join(args.res, "low_freq_group_tags_report.tsv"), sep='\t', index=False)

    # does alpha deteriorate for infrequent labels
    rare_analysis = (
        tag_iaa_res
        .assign(
            freq_band=pd.cut(
                tag_iaa_res["tag_freq"],
                bins=[0, 5, 10, 20, 50, np.inf],
                labels=["1-5", "6-10", "11-20", "21-50", "51+"],
            )
        )
        .groupby("freq_band", observed=False)  # include empty bins
        .agg(
            n_tags=("variable", "size"),
            mean_alpha=("alpha", "mean"),
            median_alpha=("alpha", "median"),
            min_alpha=("alpha", "min"),
        )
    )

    print("\nAnalysis of IAA alpha by tag frequency band:")
    print(rare_analysis)

    # ---- GOLD STANDARD: majority voting for triple annotated items ----
    # replace human-readable values with category-value codes using the scheme map
    # For binary variables, majority vote is "yes" if at least 2 annotators said "yes", otherwise "no".

    # majority vote and disagreement score for each sent_id
    gold_rows = []

    for sent_id, group in cross_annotated.groupby("sent_id"):
        gold_row = adjudicate_id(group)

        gold_row["disagreement"] = count_disagreement(group, method="strict")

        gold_rows.append(gold_row)

    gold_df = pd.DataFrame(gold_rows)

    # save adjudicate ids to send these items for revision
    to_adjudicate = gold_df.loc[
        gold_df["adjudicate"] == "yes", ["annotator", "sent_id", "left_context", "text", "right_context",
                                         "further_context", "date", "dataset", "year"]]

    to_adjudicate.to_csv(os.path.join(args.reworks, "adjudicate_them.tsv"), sep="\t", index=False)

    # drop rows where incomplete or adjudicate = "yes" for the gold standard, as these need to be revised by annotators
    gold_df_out = gold_df.loc[gold_df["adjudicate"] == "no"].reset_index(drop=True)

    gold_df_out = gold_df_out.drop(['adjudicate', 'incomplete', 'dataset', 'annotation_id'], axis=1)

    print(f"\nGold dataset with {len(gold_df_out)} rows and {len(gold_df_out.columns)} columns:")
    print(f"\t--- after removing {len(to_adjudicate)} adjudication-needed items ---")
    print(gold_df_out.columns.tolist())

    print("\nDisagreement range:", gold_df_out["disagreement"].min(), "-", gold_df_out["disagreement"].max())
    print("Mean:", round(gold_df_out["disagreement"].mean(), 2))
    gold_df_out.to_csv(os.path.join(args.gold_to, f"{my_date}_gold_dataset.tsv"), sep='\t', index=False)

    # debugging
    # temp = gold_df[(gold_df["group_appealed"] == "NA") & (gold_df["group_mentioned"] == "yes")]
    # print(temp[["sent_id", "group_mentioned", "group_appealed", "group_tags"]].head())
    # print(temp.shape)
    # exit()

    # --- Additional statistics --
    # statistics on binary variables: how many "yes" items per variable
    base_binary_variables = [
        'group_mentioned',
        'group_appealed',
        'intersectional',
        'multiple_groups',
        'opposed_groups',
        'pejorative',
        'adjudicate',
    ]

    failed_adjudications = (gold_df["adjudicate"] == "yes").sum()

    bases = ['triple', "gold"]
    datas = [cross_annotated, gold_df_out]

    for base, anno_df in zip(bases, datas):
        variables = [v for v in base_binary_variables if v in anno_df.columns]

        print(f"\n{base.upper()} ANNOTATED ITEMS")

        # Construct the summary table
        binary_summary = pd.DataFrame(
            [
                {
                    "variable": var,
                    "n_items": (
                        anno_df
                        .groupby("sent_id")[var]
                        .apply(lambda x: (x == "yes").any())
                        .sum()
                    ),
                }
                for var in variables
            ]
        )

        # Gold: remove adjudication row
        if base == "gold":
            binary_summary = binary_summary[
                binary_summary["variable"] != "adjudicate"
                ]
            total_items = anno_df["sent_id"].nunique()  # anno_df["sent_id"].nunique() - failed_adjudications

        else:  # triple
            binary_summary = pd.concat(
                [
                    binary_summary,
                    pd.DataFrame(
                        [{
                            "variable": "failed adjudication",
                            "n_items": failed_adjudications,
                        }]
                    ),
                ],
                ignore_index=True,
            )
            total_items = anno_df["sent_id"].nunique()

        # Prepend total row
        binary_summary = pd.concat(
            [
                pd.DataFrame(
                    [{
                        "variable": f"Total ({base})",
                        "n_items": total_items,
                    }]
                ),
                binary_summary,
            ],
            ignore_index=True,
        )

        print(binary_summary)

        binary_summary.to_csv(os.path.join(args.res, f"yes_counts_{base}.tsv"), sep='\t', index=False)
        plot_binary_summary(binary_summary, save_as=os.path.join(args.pics, f"yes_counts_{base}.png"), show=True)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
