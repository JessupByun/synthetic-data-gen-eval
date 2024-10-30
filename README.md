# synthetic-data-gen-eval

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation and Setup](#installation-and-setup)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Create a Virtual Environment and Install Dependencies](#2-create-a-virtual-environment-and-install-dependencies)
   - [3. Set Up Environment Variables for API Access Keys](#3-set-up-environment-variables-for-api-access-keys)
4. [Usage](#usage)
   - [Running the Model Deployment Script](#running-the-model-deployment-script)
   - [Running Evaluation Tests](#running-evaluation-tests)
5. [Models Deployed](#models-deployed)
6. [Real Tabular Datasets Used](#real-tabular-datasets-used)
7. [References](#references)
8. [In Progress](#in-progress)

## Project Overview

- **Undergraduate Researcher:** Jessup Byun, UCLA
- **Project Start Date:** September 2024
- **Research Mentor:** Xiaofeng Lin, PhD Statistics, UCLA  
- **Faculty Advisor:** Prof. Guang Cheng, UCLA Trustworthy AI Lab
---

This repository contains the codebase for my independent research conducted at the **UCLA Trustworthy AI Lab** during **Fall '24**. My research focuses on synthetic tabular data generation and evaluation using advanced models like **LLaMA** and **TabSyn** deployed through frameworks like the **Groq API**. This work primarily assesses the tradeoff between utility and privacy in synthetic data generation.

⚠️ **Note**: Some code and evaluation components, particularly those involving lab-specific methods, may be omitted or simplified to maintain security and lab privacy standards. 

## Repository Structure

The repository is organized as follows:

- **Model Deployment (`model_deployment.py`)**: Contains code for deploying models via the Groq API for synthetic data generation. Model prompts and generation parameters can be customized in this file.
- **Evaluation Scripts**: Scripts to evaluate utility and privacy metrics on generated data using custom lab-developed frameworks. Sensitive evaluation metrics and some proprietary code are excluded.
- **Sample Datasets (`sample_datasets/`)**: Includes real-world datasets used as samples for synthetic data generation and baselines for evaluation. Datasets in this folder are referenced in code as part of the generation and evaluation processes.
- **`requirements.txt`**: Lists dependencies needed to run the project. Excludes sensitive packages or proprietary libraries.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JessupByun/synthetic-data-gen-eval.git
cd synthetic-data-gen-eval
```

### 2. Create a Virtual Environment and Install Dependencies

#### Create a Virtual Environment with Python 3.10:

*Note: The repository uses a virtual environment with Python 3.10 due to incompabilities with evaluation libraries when using newer versions of Python*

```bash
python3.10 -m venv .venv
```

#### Activate the virtual environment:

- macOS/Linux:
```bash
source .venv/bin/activate
```
- Windows:
```bash
.venv\Scripts\activate
```

#### Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables for API Access Keys

- Create a .env file in the root of your directory
- Add your Groq API Key:
```bash
GROQ_API_KEY="your_api_key_here"
```

## Usage

### Running the Model Deployment Script

To deploy models for synthetic data generation, enter the folder of the sample dataset and run:
```bash
{folder_name}_model_deployment.py
```

### Running Evaluation Tests

Evaluations for this project primarily assess the utility of generated synthetic data, but can also evaluate fidelity and privacy. Due to lab privacy requirements, some evaluation metrics and methods may not be included in this repository. To run evaluation scripts on fidelity, utility, and privacy:
```bash
{folder_name}_evaluation.py
```
Once available, it may also be possible to run another script that uses the proprietary "Alfred_Analytica" evaluation library developed by the UCLA Trustworthy AI Lab

## Models Deployed
Specific models deployed via the Groq API include:
- LLaMA 3.1 70B
- LLaMA 3.1 8B
- LLaMA 3.2 1B

## Real Tabular Datasets Used
The following real tabular datasets are used as baselines for evaluation, most of which are sourced from Kaggle:

- **Insurance** (Kaggle): [Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## References

[1] The prompt structure used for synthetic data generation is adapted from example B.5 of the research paper: *Curated LLM: Synergy of LLMs and Data Curation for Tabular Augmentation in Low-Data Regimes* by Seedatk, Huynh, et al. The reference is specifically applied in the data generation prompt structure within this codebase to emulate realistic and diverse synthetic samples. For further details, refer to the full paper [here](https://arxiv.org/pdf/2312.12112).

[2] The list of real tabular datasets used in this research is adapted from Appendix A of the paper: *AutoDiff: Diffusion-based Generative Models for Tabular Data Synthesis*, as detailed by the authors. This list has been integrated to ensure comprehensive evaluation across diverse dataset types. For more information, please see the full paper [here](https://arxiv.org/pdf/2310.15479).

## In Progress:

#### This project is currently evolving to include advanced methods for synthetic data generation and evaluation. Planned tasks include:

**Exploring Diffusion-Based Models:** Building on the initial phase with large language models (LLMs), I plan to experiment with latent diffusion models, which are known for their capability in generating high-dimensional data. These models may provide enhanced fidelity and diversity in synthetic datasets.

**Refining Utility Evaluation Metrics:** The project will incorporate new metrics for evaluating the utility of synthetic data, focusing on the utility-privacy tradeoff. This includes a comprehensive evaluation using composite utility scores and privacy assessment protocols.

**Broadening Model Testing Scope:** Besides Groq API-deployed LLMs like LLaMA, future iterations will explore additional experimental and state-of-the-art generative models, benchmarking their performance against current models.

**Improving Reproducibility and Scalability:** Future updates aim to refine the codebase for easier deployment and reproducibility, especially around evaluation frameworks, to enable scalability across various synthetic data applications.

