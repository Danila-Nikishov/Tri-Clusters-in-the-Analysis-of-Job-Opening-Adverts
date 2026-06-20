# Triclustering Analysis of Job Vacancy Requirements, Responsibilities and Conditions

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Tokenization-yellow.svg)
![Clustering](https://img.shields.io/badge/Triclustering-Research-green.svg)
![Status](https://img.shields.io/badge/Status-Research%20Project-success.svg)

## Overview

This repository contains a complete research pipeline for analyzing job vacancies collected from **HH.ru**. The project focuses on discovering latent relationships between three key components of vacancy descriptions:

- **Requirements** (skills, technologies, qualifications)
- **Responsibilities** (tasks and duties)
- **Conditions** (benefits, working conditions, compensation)

The proposed approach represents vacancies as a **three-dimensional relevance tensor** and applies **triclustering** to identify coherent groups of requirements, responsibilities, and conditions that frequently occur together.

The repository includes data collection, NLP preprocessing, relevance estimation, tensor construction, and triclustering using the **Tricluster Box** algorithm.

---

## Repository Structure

```text

Tri-Clusters-in-the-Analysis-of-Job-Opening-Adverts
├── 1_hh_parsing.ipynb
├── 2_gaps_estimate.ipynb
├── 3_creating_tokens.ipynb
├── 4_tokens_processing.ipynb
├── 5_corelevance_matrix.ipynb
├── 6_clusterization.ipynb
├── three_cluster_box.py
└── README.md
```

---

## Pipeline

### 1. Vacancy Collection

**File:** `1_hh_parsing.ipynb`

Downloads vacancy data from the HH.ru API and stores the raw vacancy descriptions for further processing.

Main tasks:

- Vacancy search
- Vacancy metadata collection
- Description extraction
- Dataset creation

---

### 2. Dataset Preparation and Quality Assessment

**File:** `2_gaps_estimate.ipynb`

Performs additional processing of the collected dataset and evaluates data completeness.

Main tasks:

- Missing value analysis
- Dataset quality assessment
- Filtering incomplete vacancies
- Preparing data for NLP processing

---

### 3. Token Extraction

**File:** `3_creating_tokens.ipynb`

Extracts meaningful tokens from vacancy texts and separates them into three semantic categories:

- Requirements
- Responsibilities
- Conditions

---

### 4. Token Processing and Relevance Matrices

**File:** `4_tokens_processing.ipynb`

Constructs three two-dimensional relevance matrices using AST (annotated suffix trees) approach:

1. **Vacancy × Requirement**
2. **Vacancy × Responsibility**
3. **Vacancy × Condition**

These matrices describe the relationship between vacancy texts and extracted semantic tokens.

---

### 5. Three-Dimensional Corelevance Tensor Construction

**File:** `5_corelevance_matrix.ipynb`

Builds a three-dimensional relevance tensor:

```text
Requirement × Responsibility × Condition
```

The tensor values are calculated using a generalized version of the **Kulczynski coefficient** extended to the three-dimensional case.

This stage captures higher-order associations that cannot be observed using pairwise analysis alone.

---

### 6. Triclustering

**File:** `6_clusterization.ipynb`

Implements the complete triclustering workflow using the **Tricluster Box** algorithm.

Main tasks:

- Tensor loading
- Tricluster extraction
- Cluster quality evaluation
- Interpretation of resulting clusters

The output consists of groups of:

- related requirements,
- corresponding responsibilities,
- associated working conditions.

---

## Tricluster Box Algorithm

**File:** `three_cluster_box.py`

This is the only standalone Python module in the repository.

The file contains an implementation of the **Tricluster Box** algorithm used to discover dense substructures in the three-dimensional relevance tensor.

The algorithm identifies coherent triples:

```text
(Requirements, Responsibilities, Conditions)
```

that frequently co-occur across vacancies.

---

## Methodology

The overall workflow can be represented as:

```text
HH.ru Vacancies
        │
        ▼
 Text Collection
        │
        ▼
 Token Extraction
        │
        ▼
 Vacancy-Token Matrices
        │
        ▼
 3D Corelevance Tensor
        │
        ▼
 Tricluster Box
        │
        ▼
 Interpretable Vacancy Patterns
```

---

## Research Motivation

Traditional vacancy analysis usually focuses on isolated skills or keyword frequencies.

This project explores a richer representation by simultaneously analyzing:

- what employers require,
- what employees are expected to do,
- what employers offer in return.

The resulting triclusters provide a compact representation of labor market patterns and can be used for:

- labor market analysis,
- skill ecosystem discovery,
- job family identification,
- vacancy recommendation systems,
- HR analytics.

---
