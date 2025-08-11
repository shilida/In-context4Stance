
# An Empirical Study of LLMs via In-Context Learning for Stance Classification

This repository contains the official code and resources for our paper: **"An empirical study of LLMs via in-context learning for stance classification"**.

> **Abstract:** The rapid advancement of large language models (LLMs) creates new research opportunities in stance classification. However, existing studies often lack a systematic evaluation and empirical analysis of the performance of mainstream large models. In this paper, we systematically evaluate the performance of 5 SOTA large language models, including LLaMA, DeepSeek, Qwen, GPT, and Gemini, on stance classification using 13 benchmark datasets. We explore the effectiveness of two strategies — random selection and semantic similarity selection — within the framework of in-context learning. By comparing these approaches through cross-domain and in-domain experiments, we reveal how they impact model performance and provide insights for future optimization. Overall, this study clarifies the influence of different models and sampling strategies on stance classification performance and suggests directions for further research. Our code is available at: https://github.com/shilida/In-context4Stance.

## Table of Contents
- [Setup](#setup)
- [Data](#data)
  - [Data Acquisition](#data-acquisition)
  - [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
  - [1. Generate In-Context Learning Samples](#1-generate-in-context-learning-samples)
  - [2. Run In-Domain Stance Classification](#2-run-in-domain-stance-classification)
  - [3. Run Cross-Domain Stance Classification](#3-run-cross-domain-stance-classification)
- [Citation](#citation)

## Setup
 Install the required Python packages. 
    The main dependencies are:
    - `langchain`
    - `langchain-openai`
    - `tweet-preprocessor`
    - `pandas`
    - `pydantic`
    - `scikit-learn`
    - `tqdm`
    - `wandb`

## Data

### Data Acquisition

The datasets used in our study can be obtained from their original sources. We are grateful to the creators for making their work public.

*   **ARC, ArgMin, FNC-1, IAC, IBM-CS, Perspectrum, SemEval-2016T6**: [UKPLab/mdl-stance-robustness](https://github.com/UKPLab/mdl-stance-robustness#preprocessing)
*   **VAST**: [emilyallaway/zero-shot-stance](https://github.com/emilyallaway/zero-shot-stance)
*   **PolDeb**: [Political Debates Corpus](http://mpqa.cs.pitt.edu/corpora/political_debates/)
*   **Emergent**: [Dropbox Link](https://www.dropbox.com/scl/fo/bn26dd72tz6f3bj5pa5ix/ABwyIVSFvxwk4vse9YD_32s/emergent?rlkey=m7mkfayakm5s126rtbaskg2eq&e=1)
*   **NLPCC2016**: [NLPCC 2016 Shared Task](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html)
*   **COVID-19**: [liviucotfas/covid-19-vaccination-stance-detection](https://github.com/liviucotfas/covid-19-vaccination-stance-detection)
*   **ImgArg**: [ImageArg Shared Task](https://github.com/ImageArg/ImageArg-Shared-Task)

We also thank the authors of [checkstep/senti-stance](https://github.com/checkstep/senti-stance) for their contributions to the field.

### Data Preprocessing

The scripts for preprocessing the datasets are located in the `process_dataset/` directory. Please refer to the scripts within that folder for details on how to process each dataset.

## Usage

Before running the experiments, please ensure you have configured your API keys and any model-specific parameters within the scripts.

### 1. Generate In-Context Learning Samples

To generate the demonstration samples for in-context learning based on semantic similarity, run the following script:

```bash
python get_sim_samples.py
```

### 2. Run In-Domain Stance Classification

To run the main experiments for in-domain stance classification:

```bash
python main.py
```

### 3. Run Cross-Domain Stance Classification

To run the experiments for cross-domain stance classification:

```bash
python main_cross_domain.py
```

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{SHI2026104322,
    title = {An empirical study of LLMs via in-context learning for stance classification},
    journal = {Information Processing & Management},
    volume = {63},
    number = {1},
    pages = {104322},
    year = {2026},
    issn = {0306-4573},
    doi = {https://doi.org/10.1016/j.ipm.2025.104322},
    url = {https://www.sciencedirect.com/science/article/pii/S0306457325002638},
    author = {Lida Shi and Fausto Giunchiglia and Ran Luo and Daqian Shi and Rui Song and Xiaolei Diao and Hao Xu},
    keywords = {Stance classification, In-context learning, LLMs},
    abstract = {The rapid advancement of large language models (LLMs) creates new research opportunities in stance classification. However, existing studies often lack a systematic evaluation and empirical analysis of the performance of mainstream large models. In this paper, we systematically evaluate the performance of 5 SOTA large language models, including LLaMA, DeepSeek, Qwen, GPT, and Gemini, on stance classification using 13 benchmark datasets. We explore the effectiveness of two strategies — random selection and semantic similarity selection — within the framework of in-context learning. By comparing these approaches through cross-domain and in-domain experiments, we reveal how they impact model performance and provide insights for future optimization. Overall, this study clarifies the influence of different models and sampling strategies on stance classification performance and suggests directions for further research. Our code is available at: https://github.com/shilida/In-context4Stance.}
}
```