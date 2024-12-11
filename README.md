# synthetic-data-gen-eval

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation and Setup](#installation-and-setup)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Create a Virtual Environment and Install Dependencies](#2-create-a-virtual-environment-and-install-dependencies)
   - [3. Set Up Environment Variables for API Access Keys](#3-set-up-environment-variables-for-api-access-keys)
4. [Models Deployed](#models-deployed)
5. [Real Tabular Datasets Used](#real-tabular-datasets-used)
6. [References](#references)
7. [In Progress](#in-progress)

## Project Overview

- **Undergraduate Researcher:** Jessup Byun, UCLA  
- **Project Start Date:** September 2024  
- **Research Mentor:** Xiaofeng Lin, PhD Statistics, UCLA  
- **Faculty Advisor:** Prof. Guang Cheng, UCLA Trustworthy AI Lab  

---

This repository contains the codebase for my independent research conducted at the **UCLA Trustworthy AI Lab** during **Fall '24**. My research focuses on synthetic tabular data generation and evaluation using advanced models like **LLaMA** and **Mixtral** deployed through frameworks like the **Groq API**. This work assesses the tradeoff between utility and privacy in synthetic data generation.

⚠️ **Note**: Some code and evaluation components, particularly those involving lab-specific methods, may be omitted or simplified to maintain security and lab privacy.

## Repository Structure

- **`evaluation/`**: Contains evaluation scripts for each dataset (t-sne plot generation). Each test dataset may have its own subdirectories for evaluation results
- **`model_deployment/`**: Scripts for generating synthetic data for each test case, named as `dataset_name_model_deployment.py`.
- **`test_data/`**: Stores real and synthetic datasets for testing purposes:
  - **`real_data/`**: Directory for real datasets, with files like `dataset_name.csv`.
  - **`synthetic_data/`**: Directory for synthetic datasets, with files like `dataset_name_synthetic_data.csv`.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JessupByun/synthetic-data-gen-eval.git
cd synthetic-data-gen-eval
```

### 2. Create a Virtual Environment and Install Dependencies

#### Create a Virtual Environment with Python 3.10

*Note: The repository uses Python 3.10 due to incompatibilities with evaluation libraries when using newer versions.*

```bash
python3.10 -m venv .venv
```

#### Activate the Virtual Environment

- macOS/Linux:
```bash
source .venv/bin/activate
```
- Windows:
```bash
.venv\Scripts\activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables for API Access Keys

- Create a `.env` file in the root directory.
- Add your Groq API Key:
```bash
GROQ_API_KEY="your_api_key_here"
```

## Models Deployed

Specific models deployed via the Groq API include:
- **LLaMA 3.1**: 70B, 8B
- **LLaMA 3.2**: 1B
- **Mixtral**: 8x7B

## Real Tabular Datasets Used

The following real tabular datasets are used as baselines for evaluation:
- **Insurance** (Kaggle): [Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **combined_df_cty_week2.csv** (Private)
- **labMergeDemo1Trimmed.csv** (Private)

## References

[1] The prompt structure used for synthetic data generation is adapted from example B.5 of the research paper: *Curated LLM: Synergy of LLMs and Data Curation for Tabular Augmentation in Low-Data Regimes* by Seedatk, Huynh, et al. For further details, refer to the full paper [here](https://arxiv.org/pdf/2312.12112).

[2] The list of real tabular datasets is adapted from Appendix A of the paper: *AutoDiff: Diffusion-based Generative Models for Tabular Data Synthesis*. For more information, see the full paper [here](https://arxiv.org/pdf/2310.15479).

## In Progress

**Enhancements Planned:**
- Experimenting with latent diffusion models to improve synthetic data quality.
- Experimenting with smarter and bigger LLMs.
- Expanding datasets to include more diverse domains.
- Refining evaluation metrics for fidelity, utility, diversity, privacy
- Preparing a manuscript for publication to share findings with the broader research community.
