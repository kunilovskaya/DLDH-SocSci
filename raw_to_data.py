"""
20 June 2026
Maria Kunilovskaya

This module contains functions for post-processing the output of Label Studio, a data labeling tool.

17 Jun 2026
Using a tsv file with annotated items from Label Studio export:
-- make sense of the structure to see how each task is represented in the output
-- get quantitative desctiption of the annptated data (e.g. how many items, how many labels, etc.)
-- describe performance by annotator (how many tasks were done by each)
-- how many tasks are cross-annotated,
-- prepare the input for IAA and gold standard creation

    # these are surface human-readable annotation tasks for student_groups
    # Task 1: Is a social group invoked as part of a political argument? <-- two negative alternatives; None==yes
    # Task 2: Which social group is invoked in the argument? 12 multiple choice + other_group flag
    # Task 3: How is the group invoked in the argument? What is the nature of the group appeal? <-- 3 single multiple choice, 1 multi-multiple choice

USAGE:
set the required RUN parameter RUN = "trial_student_groups"  # "main_student_groups"
python3 raw_to_data.py

"""

import argparse
import json
import os
import sys
import time

from datetime import datetime

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def tag_coverage_summary(df_scope, ordering_df, verbose=False):
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
        ordering_df.loc[
            ordering_df["category_order"].between(10, 199),
            "tag",
        ]
    )

    appeal_inventory = set(
        ordering_df.loc[
            ordering_df["category_order"] >= 200,
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
            ordering_df.loc[ordering_df["tag"].isin(unseen_tags)]
            .sort_values(["category_order", "group_order"])
        )
        print("Unseen tags:")
        print(
            unseen_df[["category", "group"]]
            .to_string(index=False)
        )

        coverage = (
            ordering_df.assign(
                seen=ordering_df["tag"].isin(seen_tags)
            )
            .groupby(["category_order", "category"], as_index=False)
            .agg(
                total=("tag", "size"),
                seen=("seen", "sum")
            )
        )

        coverage["unseen"] = coverage["total"] - coverage["seen"]
        coverage["unseen_pct"] = 100 * coverage["unseen"] / coverage["total"]

        coverage = coverage.sort_values("category_order").round(1)
        print("\nTag coverage by category:")
        print(
            coverage[["category", "total", "seen", "unseen", "unseen_pct"]]
            .to_string(index=False)
        )

    return pd.DataFrame(rows).set_index("category")


def plot_binary_summary(binary_summary, save_as=None, show=True):
    """
    Plot counts of items with at least one 'yes' annotation.

    Parameters
    ----------
    binary_summary : pd.DataFrame
        Columns: ['variable', 'n_items']
    save_as : str | Path | None
        Path to save figure. If None, only displays.
    title : str
        Figure title.
    figsize : tuple
        Matplotlib figure size.
    """

    plot_df = binary_summary.sort_values(
        "n_items",
        ascending=False
    ).reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    figsize = (8, 4)
    fig, ax = plt.subplots(figsize=figsize)

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
            os.makedirs(i, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logs}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
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


# main_student_groups
VALUE_TO_GROUP_TAG_NESTED = {
    "age": {
        "unborn": "AGE_UNBORN",
        "pre-school": "AGE_PRESCH",
        "school age": "AGE_SCHOOL",
        "children": "AGE_CHILDREN",
        "teenagers": "AGE_TEENAGERS",
        "young adults": "AGE_YOUNG",
        "students": "AGE_IN_EDUCATION",
        "mature people": "AGE_MATURE",
        "seniors": "AGE_SENIOR",
        "retired": "AGE_RETIRED",
        "other": "AGE_AGE",
    },
    "gender_sexuality": {
        "women": "GEN_WOMEN",
        "men": "GEN_MEN",
        "heterosexual": "GEN_SOGI_HET",
        "LGBTIQ+": "GEN_SOGI_LGB",
        "gender diverse": "GEN_DIVERSE",
    },
    "family": {
        "mothers": "FAM_MOTHER",
        "fathers": "FAM_FATHER",
        "single parents": "FAM_SINGLE_PARENTS",
        "childless": "FAM_CHILDLESS",
        "parents": "FAM_PARENT",
        "grandparents": "FAM_GRAND",
        "large families": "FAM_LARGE",
        "families": "FAM_FAM",
    },
    "disability": {
        "disabled people": "DIS_DIS",
        "other health-related group": "DIS_OTHER",
    },
    "ethnicity": {
        "majority/national in-group": "ETH_MAJORITY",
        "named origin group": "ETH_ORIGIN_GROUP",
        "ethnic minority": "ETH_MIN",
    },
    "migration": {
        "refugees": "MIG_REFUGEE",
        "descendants": "MIG_DESCENDANT",
        "resettler": "MIG_RESETTLER",
        "undocumented": "MIG_UNDOC",
        "immigrants": "MIG_IMMIGRANT",
    },
    "citizenship": {
        "citizens": "CIT_CITIZEN",
        "non-citizens": "CIT_NONCITIZEN",
        "naturalized persons": "CIT_NATURALIZED",
        "other status": "CIT_OTHER",
    },
    "religion": {
        "secular": "REL_SECULAR",
        "christian": "REL_CHRIST",
        "muslim": "REL_MUSLIM",
        "jewish": "REL_JEW",
        "other religion": "REL_OTHER",
    },
    "geography": {
        "urban": "GEO_URBAN",
        "rural": "GEO_RURAL",
        "region-specific": "GEO_REGION",
        "affluent regions": "GEO_AFFLUENT",
        "disadvantaged regions": "GEO_DISADV",
        "other regional groups": "GEO_OTHER",
    },
    "education": {
        "highly educated": "EDU_HIGH",
        "mid-level": "EDU_MEDIUM",
        "low-educated": "EDU_LOW",
        "in education": "EDU_IN_EDUCATION",
        "other education-related group": "EDU_OTHER",
    },
    "income": {
        "rich/wealthy": "INC_RICH",
        "middle class": "INC_MIDDLE",
        "poor/low-income": "INC_POOR",
        "working class": "INC_WORKING",
        "unemployed": "INC_INSECURE",
        "welfare recipients": "INC_WELFARE",
        "other income-related group": "INC_OTHER",
    },
    "occupation": {
        "large employers & self-employed professionals": "OCC_OESCH_1",
        "small business owners": "OCC_OESCH_2",
        "technical (semi)-professionals": "OCC_OESCH_3",
        "production workers": "OCC_OESCH_4",
        "(associate) managers": "OCC_OESCH_5",
        "office clerks": "OCC_OESCH_6",
        "socio-cultural (semi)-professionals": "OCC_OESCH_7",
        "service workers": "OCC_OESCH_8",
        "employed": "OCC_EMPLOYED",
    },
    "other_groups": {
        "non-RISS": "OTH_NON_RISS",
    },
}

TRIAL_VALUE_TO_GROUP_TAG_NESTED = {
    "age": {
        "unborn": "AGE_UNBORN",
        "pre-school": "AGE_PRESCH",
        "school age": "AGE_SCHOOL",
        "children": "AGE_CHILD",
        "young adults": "AGE_YOUNGAD",
        "mature people": "AGE_MATURE",
        "seniors": "AGE_SENIOR",
        "retired": "AGE_RETIRED",
    },

    "gender_sexuality": {
        "men": "GEN_MEN",
        "women": "GEN_WOMEN",
        "transgender": "GEN_TRANS",
        "gender-diverse": "GEN_DIVERSE",
        "non-binary": "GEN_NONBINARY",
        "heterosexual": "GEN_SOGI_HET",
        "LGBT(+)": "GEN_LGBT",
    },

    "family": {
        "mothers": "FAM_MOTHERS",
        "fathers": "FAM_FATHERS",
        "single parents": "FAM_SINGLE_PARENTS",
        "childless": "FAM_CHILDLESS",
        "single mothers": "FAM_SINGLE_MOTHERS",
        "parents": "FAM_PARENTS",
        "grandparents": "FAM_GRANDPARENTS",
        "families": "FAM_FAMILIES",
        "large families": "FAM_LARGE_FAMILIES",
    },

    "disability": {
        "disabled people": "DIS_DISABLED",
        "chronic illness": "DIS_CHRONIC",
        "neurodivergent": "DIS_NEURODIV",
    },

    "ethnicity": {
        "ethnic minority": "ETH_MINORITY",
        "named origin group": "ETH_NAMED",
    },

    "migration": {
        "immigrants": "MIG_IMMIGRANTS",
        "refugees": "MIG_REFUGEES",
        "descendants": "MIG_DESCENDANTS",
        "resettler": "MIG_RESETTLERS",
        "majority": "MIG_MAJORITY",
        "undocumented": "MIG_UNDOCUMENTED",
        "non-citizens": "MIG_CIT_NON_CITIZENS",
        "naturalized persons": "MIG_CIT_NATURALIZED",
    },

    "religion": {
        "secular": "REL_SECULAR",
        "christian": "REL_CHRISTIAN",
        "muslim": "REL_MUSLIM",
        "jewish": "REL_JEWISH",
        "other religion": "REL_OTHER",
    },

    "geography": {
        "urban": "GEO_URBAN",
        "rural": "GEO_RURAL",
        "region-specific": "GEO_REGION",
        "affluent regions": "GEO_AFFLUENT",
        "disadvantaged regions": "GEO_DISADV",
    },

    "education": {
        "educated": "EDU_HIGH",
        "mid-level": "EDU_MID",
        "low-educated": "EDU_LOW",
        "in education": "EDU_IN_EDU",
        "skilled": "EDU_SKILLED",
        "unskilled": "EDU_UNSKILLED",
    },

    "income": {
        "rich/wealthy": "INC_RICH",
        "middle class": "INC_MIDDLE_CLASS",
        "poor/low-income": "INC_LOW_INCOME",
        "working class": "INC_WORKING_CLASS",
        "precarious/unemployed": "INC_PRECARIOUS",
        "welfare recipients": "INC_WELFARE_RECIPIENTS",
    },

    "occupation": {
        "managers/executives": "OCC_MANAGERS",
        "professionals": "OCC_PROFESSIONALS",
        "semi-professionals": "OCC_SEMIPROFESSIONALS",
        "workers": "OCC_WORKERS",
        "service workers": "OCC_SERVICE_WORKERS",
        "employed": "OCC_EMPLOYED",
        "public sector": "OCC_PUBLIC_SECTOR",
        "agriculture": "OCC_AGRICULTURE",
        "arts": "OCC_ARTS",
        "extractive": "OCC_EXTRACTIVE",
        "healthcare": "OCC_HEALTHCARE",
        "educators": "OCC_EDUCATORS",
        "self-employed": "OCC_SELF_EMPLOYED",
    },
    "other_groups": {
        "non-RISS": "OTH_NON_RISS",
    },
}


VALUE_TO_APPEAL_TAG_NESTED = {
    "frame_flags": {
        "economic": "REASONING_ECONOMIC",
        "cultural": "REASONING_CULTURAL",
        "both": "REASONING_BOTH",
        "unclear": "REASONING_UNCLEAR"
    },
    "polarity_flags": {
        "pro-group": "POLARITY_PRO_GROUP",
        "anti-group": "POLARITY_ANTI_GROUP",
        "mixed": "POLARITY_MIXED",
        "unclear": "POLARITY_UNCLEAR"
    },
    "stance_flags": {
        "endorse": "STANCE_ENDORSE",
        "reject": "STANCE_REJECT",
        "partial": "STANCE_PARTIAL",
        "unclear": "STANCE_UNCLEAR"
    }
}

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
    "REASONING": "reasoning",
    "POLARITY": "polarity",
    "STANCE": "stance"
}


def throughput(df, available_str=None):  # return a df with columns: parameter, hansard (1980, 2025, 2026)
    total_items = df["sent_id"].nunique()
    unique_annotations = df["annotation_id"].nunique()
    triple_or_more_annotated_pct = df.groupby("sent_id")["annotation_id"].nunique().ge(3).mean() * 100
    double_annotated_pct = df.groupby("sent_id")["annotation_id"].nunique().eq(2).mean() * 100
    annotation_counts = (
        df.groupby("sent_id")["annotation_id"]
        .nunique()
    )
    triple_or_more_annotated_num = (annotation_counts >= 3).sum()
    double_annotated_num = (annotation_counts == 2).sum()

    total_annotators = df["annotator"].nunique()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    annotation_start = (
        df["created_at"].min()
        .tz_convert("Europe/Berlin")
    )

    annotation_end = (
        df["updated_at"].max()
        .tz_convert("Europe/Berlin")
    )

    project_start_str = annotation_start.strftime("%Y-%m-%d %H:%M")
    project_end_str = annotation_end.strftime("%Y-%m-%d %H:%M")

    res_dict = {
        "Available items": available_str,
        # appeal detection, group classification, inc intersections,
        # reasoning, polarity, stance,
        # multiple groups, opposed groups, pejorative
        "Decision per item": 10,
        "Annotated items": int(total_items),
        "Triple+ annotated": f"{triple_or_more_annotated_num} ({triple_or_more_annotated_pct.round(2)} %)",
        "Double-annotated": f"{double_annotated_num} ({double_annotated_pct.round(2)} %)",
        "Annotations": int(unique_annotations),
        "Annotators": int(total_annotators),
        "Comments": df["scheme_feedback"].notna().sum(),
        "Avg. time per item": (df["lead_time"].mean() / 60).round(2),
        "Project start": project_start_str,
        "Project end": project_end_str,
    }
    # transform to df with columns: parameter, hansard (1980, 2025, 2026)

    res_df = pd.DataFrame(res_dict, index=[0])

    res_df = res_df.transpose().reset_index()
    res_df.columns = ["parameter", "value"]

    return res_df


RUN = "trial_student_groups"  # "main_student_groups", "trial_student_groups"
my_date = "16June2026"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', help="", default=f'data/{RUN}/')
    parser.add_argument('--interface', help="", default=f'interface/{RUN}/interface_scheme.tsv')
    parser.add_argument('--res', default=f'res/{RUN}/')
    parser.add_argument('--pics', default=f'pics/{RUN}/')
    parser.add_argument('--logs', default=f'logs/{RUN}/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics], args=args)

    intab = [f for f in os.listdir(args.indir) if f.endswith(f"{my_date}_anon_raw.tsv")][0]
    df = pd.read_csv(args.indir + intab, sep='\t')
    
    if RUN == "trial_student_groups":
        # other strructural and non_structural are already collapsed to "non-RISS"
        avail = '60x2+120x2'
    else:  # RUN == "main_student_groups":
        # address long val in other_groups
        df["other_groups"] = df["other_groups"].replace("a social group outside RISS taxonomy", "non-RISS")
        avail = '520'
    # these are our columns of interest, we can drop the rest
    if RUN == "trial_student_groups":
        df = df[['sent_id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time',
                           'dataset', 'year', 'date', 'left_context', 'text', 'right_context', 'further_context',
                           'age', 'gender_sexuality', 'family', 'disability',
                           'ethnicity', 'migration', 'religion', 'geography',
                           'education', 'income', 'occupation',
                           'other_groups',
                           'frame_flags', 'polarity_flags', 'stance_flags', 'focus_flags',
                           'group_appeal_detection', 'scheme_feedback', 'id'
                            ]]
    else:
        df = df[['sent_id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time',
                 'dataset', 'year', 'date', 'left_context', 'text', 'right_context', 'further_context',
                 'age', 'gender_sexuality', 'family', 'disability',
                 'ethnicity', 'migration', 'citizenship', 'religion', 'geography',
                 'education', 'income', 'occupation',
                 'other_groups',
                 'frame_flags', 'polarity_flags', 'stance_flags', 'focus_flags',
                 'group_appeal_detection', 'scheme_feedback', 'id'
                 ]]

    annotated_items = throughput(df, available_str=avail)
    print("\nAnnotated items:")
    print(annotated_items)
    annotated_items.to_csv(f'{args.res}annotated_items.tsv', sep="\t", index=False)

    if RUN == "trial_student_groups":
        group_cats = ['age', 'gender_sexuality', 'family',  'disability',
                      'ethnicity', 'migration', 'religion', 'geography',
                      'education', 'income', 'occupation',
                      'other_groups']
    else:
        group_cats = ['age', 'gender_sexuality', 'family',  'disability',
                      'ethnicity', 'migration', 'citizenship', 'religion', 'geography',
                      'education', 'income', 'occupation',
                      'other_groups']

    # disentangle flags in group_appeal_detection into binary columns: group_mentioned and group_appealed
    print("\nConverting group_appeal_detection flags into binary columns...")
    print(df["group_appeal_detection"].value_counts(dropna=False))
    df["group_mentioned"] = "yes"
    df["group_appealed"] = "yes"

    df.loc[
        df["group_appeal_detection"] == "no group mentioned",
        ["group_mentioned", "group_appealed"]
    ] = ["no", "no"]

    df.loc[
        df["group_appeal_detection"] == "no group appealed",
        "group_appealed"
    ] = "no"

    # print(df[['sent_id', 'group_appeal_detection', 'group_mentioned', 'group_appealed', 'stance_flags']])
    # print(df[df['sent_id'] == 'lords_2025_33002:002'])
    # print(df.shape)

    # disentangle flags in focus_flags column into separate columns with binary values
    FLAGS = {
        "intersectional": "intersectional",
        "multiple group appeals": "multiple_groups",
        "opposed groups": "opposed_groups",
        "pejorative": "pejorative",
    }

    def extract_flags(value):
        if pd.isna(value):
            return []

        # JSON format from Label Studio
        if isinstance(value, str) and value.startswith("{"):
            try:
                return json.loads(value).get("choices", [])
            except Exception:
                return []

        # Single flag as string
        return [value]

    for flag, col in FLAGS.items():
        df[col] = df["focus_flags"].apply(
            lambda x: "yes" if flag in extract_flags(x) else "no"
        )
    cols = ["focus_flags"] + list(FLAGS.values())
    # print(df.loc[df["focus_flags"].notna(), cols].head())

    if RUN == "trial_student_groups":
        df = df.replace(TRIAL_VALUE_TO_GROUP_TAG_NESTED)
    else:
        df = df.replace(VALUE_TO_GROUP_TAG_NESTED)
    df = df.replace(VALUE_TO_APPEAL_TAG_NESTED)

    # aggregate values in multi-label group tags into new columns 'group_tags'
    df['group_tags'] = df[group_cats].values.tolist()
    # remove nan values from the lists in 'group_tags': [nan, nan, FAM_FAM, nan, nan, nan, nan ...] -> [FAM_FAM]
    df["group_tags"] = df[group_cats].apply(lambda row: [x for x in row if pd.notna(x)], axis=1)
    # print(df["group_tags"].head())

    # SANITY CHECK 1: verify or add value in intersectional column, use "yes" if len() in group_tags > 1, otherwise "no"
    df["intersectional"] = df.apply(lambda row: "yes" if len(row["group_tags"]) > 1 else "no", axis=1)

    # df['appeal_tags'] = df[['frame_flags', 'polarity_flags', 'stance_flags']].values.tolist()
    # df["appeal_tags"] = df["appeal_tags"].apply(lambda lst: [] if all(pd.isna(x) for x in lst) else lst)
    # df["appeal_tags"] = df[['frame_flags', 'polarity_flags', 'stance_flags']].apply(lambda row: [x for x in row if pd.notna(x)], axis=1)

    # SANITY CHECK 2: if there are any nans in appeal_tags lists and group_appeal_detection is "yes", put "yes" in incomplete column, else "no"
    df = df.rename(columns={'frame_flags': 'reasoning', 'polarity_flags': 'polarity', 'stance_flags': 'stance'})
    appeal_cols = ["reasoning", "polarity", "stance"]

    df["incomplete"] = df.apply(lambda row: ("yes" if row["group_appealed"] == "yes" and
                                                      row[appeal_cols].isna().any() else "no"), axis=1)

    df["incomplete"] = df.apply(lambda row: ("yes" if row["group_appealed"] == "yes" and
                                                      len(row["group_tags"]) == 0 else "no"), axis=1)

    # keep only the columns you want in the gold dataset, i.e. drop original columns, now transformed and technical columns:
    # e.g. group_appeal_detection, frame_flags, polarity_flags, stance_flags, focus_flags, 'dataset', 'scheme_feedback', 'lead_time', etc
    df = df[['annotation_id', 'annotator', 'sent_id', 'updated_at',
             'group_mentioned', 'group_appealed', 'intersectional', 'multiple_groups', 'opposed_groups', 'pejorative',
             'group_tags',
             'reasoning', 'polarity', 'stance',
             'incomplete',
             'text', 'left_context', 'right_context', 'further_context', 'year', 'date', 'dataset'
             ]]

    print(f"\nTransformed dataset (for IAA and adjudication) with {len(df)} rows and {len(df.columns)} columns:")
    print(df.head(3))
    print(df.columns.tolist())

    df.to_csv(f'{args.res}transformed_dataset.tsv', sep="\t", index=False)

    # # --- INSPECT UNSEEN TAGS ---
    interface_df = pd.read_csv(args.interface, sep='\t')
    interface_df = interface_df[~interface_df["tag"].str.contains("_*", regex=False, na=False)]
    global_coverage = tag_coverage_summary(df, interface_df, verbose=True)

    print("\nTag coverage in global annotation")
    print(global_coverage)
    global_coverage.to_csv(f'{args.res}/global_tag_coverage.csv', sep='\t', index=False)

    # --- Additional statistics --
    # statistics on binary variables: how many "yes" items per variable
    base_binary_variables = [
        'group_mentioned',
        'group_appealed',
        'intersectional',
        'multiple_groups',
        'opposed_groups',
        'pejorative',
        'incomplete'
    ]
    base = "global"

    print(f"\n{base.upper()} ANNOTATED ITEMS")
    binary_summary = pd.DataFrame(
        [
            {
                "variable": var,
                "n_items": (
                    df
                    .groupby("sent_id")[var]
                    .apply(lambda x: (x == "yes").any())
                    .sum()
                )
            }
            for var in base_binary_variables
        ]
    )
    # insert a row with base n_items
    binary_summary = pd.concat(
        [
            binary_summary,
            pd.DataFrame(
                [
                    {
                        "variable": f"Total ({base})",
                        "n_items": df["sent_id"].nunique()
                    }
                ]
            )
        ],
        ignore_index=True
    )

    print(binary_summary)
    binary_summary.to_csv(f'{args.res}/binary_yes_counts_{base}.tsv', sep='\t', index=False)

    plot_binary_summary(binary_summary, save_as=f"{args.pics}/binary_yes_counts_{base}.png")

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
