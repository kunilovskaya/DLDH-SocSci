# Evaluating LLMs as Annotators

## Materials supporting the teaching session

*Deep Learning in NLP (DLDH-SocSci)*
Groups 2 and 3 – RISS Annotation Scheme
23, 30 June 2026

UPD: 30 June 2026, 18.00

## Major changes since the original version (23 June 2026)

* Added a plug-and-play implementation for `RUN=main_student_groups` and `my_date=30June2026`, covering all **520** `sent_id`s currently available for annotation.
* Updated the anonymised raw annotation files (latest annotation: **30 June 2026, 14:11**). See `data/main_student_groups/`.
* Generated the corresponding gold dataset:

  * `data/main_student_groups/30June2026_gold_dataset.tsv`
  * Reproducible workflow:

    1. `python3 raw_to_data.py` – transforms the raw annotations and produces global annotation statistics.
    2. `python3 iaa_and_gold.py` – computes inter-annotator agreement (IAA) and generates the gold dataset.
* Added explicit handling of `NA` values for downstream annotation decisions.
* Refactored the IAA calculation into a hierarchical workflow.
* Separated IAA computation for:

  * binary and multiclass variables,
  * multilabel categories.
* Added dedicated exports for:

  * reviewer comments,
  * items containing multiple groups,
  * incomplete annotations requiring review.
  
* Fixed a number of minor issues, including:

  * interface inventories of labels and label codes,
  * team alias handling,
  * various consistency fixes.
* Substantially updated `zero-shot.ipynb` (not yet fully tested due to persistent Google Colab usage limits; further updates will follow).
* Ported the scripts for cross-platform compatibility (Linux and Windows). Thanks to everyone who reported portability issues.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kunilovskaya/DLDH-SocSci/blob/main/zero-shot.ipynb)


--- 
Materials supporting the teaching session:

**Deep Learning in NLP (DLDH-SocSci)**
Groups 2 and 3 (RISS annotation scheme)
23 June 2026

Throughout the session, the demonstration is based on RUN="trail_student_groups".
It corresponds to the outcomes of the initial annotation attempts by Group2 and Group3 separately (results are combined).

The main task is to adopt this pipeline to the outcomes of the annotation process from the current joint annotation environment [Groups2-3-on-riss-scheme (DL in NLP course)](https://app.heartex.com/projects?workspace=139506)
referred to as "main_student_groups".

## Contents

* Raw annotation exports from Label Studio (annotators anonymised, speaker-party meta removed).

* Post-processing scripts for:

  * transforming annotations into an analysis-ready dataset;
  * computing human–human inter-annotator agreement (IAA);
  * generating majority-vote gold standards.
  * label distribution analysis
  
* Scripts for generating LLM-annotations and evaluating LLMs as annotators.
* Preprocessed Hansard data for 1980, 2025, and 2026, excluding speaker and party metadata.
* Scripts for generating automatic annotations for Hansard data using LLMs.
* `trial_` and `main_student_groups` annotation schemas (as implemented in the interface, including pop-up instructions).
* 23 June 2026 Session slides.
* Example prompts and model configurations.

**zero-shot.ipynb** is a Colab notebook that demonstrates how to generate automatic annotations for the Hansard data using Qwen/Qwen3-1.7B and evaluate its performance against adjudicated human annotations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kunilovskaya/DLDH-SocSci/blob/main/zero-shot.ipynb)


## Redistribution

The materials in this repository are provided for teaching and research purposes only. Please do not redistribute datasets or annotation exports without explicit permission from the authors.

