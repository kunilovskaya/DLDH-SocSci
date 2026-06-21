# Evaluating LLMs as Annotators

Materials supporting the teaching session:

**Deep Learning in NLP**
Groups 2 and 3 (RISS annotation scheme)
23 June 2026

Throughout the sessrion, the domonstration is based on RUN="trail_student_groups".
It corresponds to the outcomes of the initial annotation attempts by Group2 and Group3 separately (results are combined).

The main task is to adopt this pipeline to the outcomes of the annotation process from the current joint annotation environment [Groups2-3-on-riss-scheme (DL in NLP course)](https://app.heartex.com/projects?workspace=139506)
referred to as "main_student_groups".

## Contents

* Raw annotation exports from Label Studio.
* Post-processing scripts for:

  * transforming annotations into analysis-ready datasets;
  * computing human–human inter-annotator agreement (IAA);
  * generating majority-vote gold standards.
  * label distribution analysis
  
* Scripts for evaluating LLMs as annotators.
* Preprocessed Hansard data for 1980, 2025, and 2026, including speaker and party metadata.
* Scripts for generating automatic annotations for Hansard data using LLMs.
* `trial_` and `main_student_groups` annotation schemas (as implemented in the interface, including pop-up instructions).
* Replication instructions.
* Session slides.
* Example prompts and model configurations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/<user>/<repo>/blob/main/notebook.ipynb)

## Redistribution

The materials in this repository are provided for teaching and research purposes only. Please do not redistribute datasets or annotation exports without explicit permission from the authors.

