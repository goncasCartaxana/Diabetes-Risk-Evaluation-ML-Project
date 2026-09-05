
# Diabetes Risk Assessment — ML Pipeline for Binary and Ternary Classification

An end-to-end machine learning pipeline that predicts diabetes risk from health survey data, comparing preprocessing and class-imbalance strategies across two classification scenarios (binary and ternary) and three model families (Random Forest, SVM, MLP).


## Context

Diabetes affects an estimated 34.2 million people in the U.S. alone, with a further ~88 million living with prediabetes and undiagnosed cases contributing to a projected $400 billion annual economic burden. Early identification of at-risk individuals is a well-studied but still challenging problem — prior work using Random Forest, SVM, and neural networks on similar survey data has consistently run into two issues: severe class imbalance in real-world diabetes datasets, and a recall/accuracy trade-off in minority-class (diabetic/prediabetic) detection.

This project builds on that prior work by applying the same three model families to the 2015 BRFSS dataset, with a specific focus on **how much preprocessing and sampling strategy — rather than model choice alone — affect the ability to correctly identify at-risk individuals.**

## Data
- **Source:** [2015 Behavioral Risk Factor Surveillance System (BRFSS)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), via Kaggle.
- **Size:** 253,680 observations, 21 input features (demographic, behavioral, and health-related — e.g. BMI, physical activity, smoking status, general health, healthcare access).
- **Two target variables, defining two classification scenarios:**
  - **Binary** (`Diabetes_binary`): No Diabetes (0) vs. Diabetes (1)
  - **Ternary** (`Diabetes_012`): No Diabetes (0), Prediabetes (1), Diabetes (2)
- **Class imbalance:** severe in both scenarios — over 86% "No Diabetes" in the binary target, and in the ternary target the "Prediabetes" class makes up only ~1.8% of samples. This imbalance is the central challenge the pipeline is built around.


## Approach

The project is split into two scenarios (binary and ternary) because the same 21 input features support two different diagnostic questions — coarse (diabetic vs. not) vs. fine-grained (including prediabetes) — and class imbalance behaves very differently in each. Splitting them lets sampling and preprocessing choices be evaluated independently per scenario rather than assuming one strategy fits both.

**Pipeline stages:**

1. **EDA** — feature distributions, class imbalance quantification, outlier detection (BMI via IQR), correlation analysis, and PCA for class-separability visualization.
2. **Preprocessing** — duplicate removal, Box-Cox transformation for skewed numerical features, standardization (StandardScaler). One-hot encoding and PCA-based dimensionality reduction were evaluated but not included in the final pipeline (see Limitations).
3. **Sampling strategies** — each scenario was tested under four conditions: no preprocessing/no sampling, preprocessing only, preprocessing + undersampling, and preprocessing + hybrid sampling (a combination of over- and undersampling).
4. **Modeling** — Random Forest, linear SVM, and MLP, each tuned via grid search (randomized search for SVM, due to runtime constraints — see Limitations), evaluated primarily on **Recall**, given the cost of missing at-risk individuals in this domain.

## Results

**Binary scenario (Recall / F1):**

| Model | No preprocessing, no sampling | Preprocessing only | + Undersampling | + Hybrid sampling |
|---|---|---|---|---|
| Random Forest | 0.162 / 0.251 | 0.164 / 0.253 | 0.631 / 0.320 | **0.687 / 0.462** |
| SVM | 0.757 / 0.445 | 0.766 / 0.446 | 0.649 / 0.388 | **0.807 / 0.438** |
| MLP | 0.177 / 0.270 | 0.180 / 0.273 | 0.590 / 0.310 | **0.804 / 0.421** |

**Ternary scenario (Recall-macro / F1-weighted):**

| Model | No preprocessing, no sampling | Preprocessing only | + Undersampling | + Hybrid sampling |
|---|---|---|---|---|
| Random Forest | 0.385 / 0.792 | 0.384 / 0.791 | 0.371 / 0.390 | **0.486 / 0.751** |
| SVM | 0.448 / 0.799 | 0.454 / 0.800 | 0.386 / 0.358 | **0.505 / 0.679** |
| MLP | 0.397 / 0.799 | 0.388 / 0.795 | 0.373 / 0.451 | **0.479 / 0.645** |

**Key findings:**
- Preprocessing alone had only a marginal effect on Recall in both scenarios.
- **Hybrid sampling consistently produced the best Recall across all models and both scenarios** — up to a **23.6% Recall improvement** for Random Forest in the binary scenario relative to the unprocessed baseline.
- Undersampling alone was inconsistent: it helped Random Forest and MLP in the binary scenario, but *hurt* every model in the ternary scenario, suggesting multi-class distinctions are more sensitive to losing majority-class data.
- SVM had the strongest Recall in both scenarios, but the highest Recall didn't always coincide with the highest F1 — a real recall/precision trade-off, more pronounced in the ternary scenario.

## Conclusions

Class-imbalance handling — specifically hybrid sampling — mattered far more than preprocessing or model choice for correctly identifying at-risk individuals, improving Recall by up to 23.6% over the unprocessed baseline. However, this comes with a real trade-off: the sampling strategy that best serves Recall isn't always the one that best serves overall F1, and this trade-off intensifies as the classification task gets more fine-grained (binary → ternary). In practice, the "best" sampling strategy is model- and scenario-dependent, not universal.

## Repository Structure
```
diabetes-risk-ml/
├── README.md
├── report.pdf # Full academic writeup
├── requirements.txt
└── notebooks/
    ├── 01_data_preparation_and_experiment_setup.ipynb # Problem definition, EDA, preprocessing, metric/hyperparameter selection & rationale; produces the dataset dictionary used below
    ├── 02_experiments_binary.ipynb # Metric/grid-search implementation, model training & evaluation — binary scenario
    └── 03_experiments_ternary.ipynb # Same as above — ternary scenario
```

Notebook `01` decides *what* metrics, models, and hyperparameters to use and produces the prepared data (across all preprocessing × sampling combinations, for both scenarios). 

Notebooks `02` and `03` each implement *how* — the actual metric functions, grid search, and model training/evaluation — for their respective scenario. 

Run `01` first; `02` and `03` depend on its output and can then be run independently of each other.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/goncasCartaxana/diabetes-risk-ml.git
cd diabetes-risk-ml
```

2. Create and activate a virtual environment (Python 3.10+ recommended):
```bash
python -m venv .venv
source .venv/bin/activate # on Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place it as instructed at the top of `01_data_preparation_and_experiment_setup.ipynb`.

5. Run `01_data_preparation_and_experiment_setup.ipynb` first, then `02_experiments_binary.ipynb` and `03_experiments_ternary.ipynb` (in either order).


## Full Report

The complete academic writeup — including background/related work, detailed EDA, methodology rationale, and full discussion — is available in [`report.pdf`](./report.pdf).

## Limitations & Future Work

- **SVM hyperparameter search was constrained by runtime.** A randomized search over a narrow parameter space was used instead of an exhaustive grid search, meaning the reported SVM results likely aren't the model's true ceiling. A more efficient or heuristic search strategy (e.g. Bayesian optimization) would allow broader exploration.
- **Dimensionality reduction and one-hot encoding were evaluated but not adopted** in the final pipeline, based on limited observed gains — this trade-off is discussed in more depth in the full report and could be revisited with different encoding/reduction strategies.
- **Future directions noted in the report:** more sophisticated resampling methods, broader hyperparameter search, and evaluation on additional imbalance-aware metrics (e.g. Precision, Specificity) beyond Recall and F1.

