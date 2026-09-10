# Rainfall Prediction in Australia: End-to-End Machine Learning Pipeline (CRISP-DM)

An end-to-end Machine Learning classification project designed to predict next-day rainfall (`RainTomorrow`) across Australian meteorological stations, strictly adhering to the **CRISP-DM** methodology.

---

## Business Problem & Context
Accurate meteorological forecasting is vital for agriculture, logistics, and resource management. The objective of this project is to build an automated, scalable classification system that predicts whether it will rain tomorrow based on current-day weather observations, while addressing target class imbalance (~78% dry days vs. ~22% rainy days).

- **Target Variable:** `RainTomorrow` (Binary: 0 = No Rain, 1 = Rain).
- **Primary Optimization Metric:** ROC-AUC and Minority-Class Recall.
- **Data Source:** [Kaggle - Australia Weather Data](https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data) (140k+ records).

---

## CRISP-DM Project Lifecycle

### 1. Business Understanding
- Frame the problem as binary classification under class imbalance.
- Define success metrics: prioritize **ROC-AUC** and **Recall** on rainy days over raw Accuracy to prevent naive majority-class bias.

### 2. Data Understanding (EDA)
- Conducted structural auditing across numeric and categorical variables.
- Detected and resolved critical **Data Leakage**: identified that `RISK_MM` directly leaks rainfall amount for the next day, necessitating its immediate removal.
- Evaluated missingness: removed features with >35% missing values (`Sunshine`, `Evaporation`, `Cloud3pm`, `Cloud9am`).

### 3. Data Preparation
- **Target Scrubbing:** Dropped records with null target values (142,193 clean rows remaining).
- **Stratified Split:** Executed an 80/20 train/test split with `stratify=y` to preserve exact class proportions (22.42% positive class).
- **Pipeline Architecture (`ColumnTransformer`):**
  - Numeric Features: Median imputation + Standard scaling (`StandardScaler`).
  - Categorical Features: Mode imputation (`most_frequent`) + One-Hot Encoding (`OneHotEncoder`, 110 features post-expansion).

### 4. Modeling & Benchmarking
Three algorithmic architectures were evaluated using class-weight balancing:
- **Baseline:** Logistic Regression (`class_weight='balanced'`).
- **Parallel Ensemble:** Random Forest Classifier.
- **Gradient Boosting:** LightGBM (`scale_pos_weight` calibrated to inverse class ratio).

| Model Architecture | ROC-AUC | Recall (Rain) | Precision (Rain) | F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8637 | 0.77 | 0.52 | 0.62 | 0.79 |
| Random Forest | 0.8625 | 0.74 | 0.54 | 0.62 | 0.80 |
| **LightGBM (Champion)** | **0.8843** | **0.79** | **0.55** | **0.64** | **0.81** |

### 5. Evaluation
- **Confusion Matrix:** LightGBM captured **78.65% of all actual rainy days** (5,014 true positives), limiting false negatives to 1,361 instances.
- **Feature Importance:** Atmospheric pressure drops at 3 PM (`Pressure3pm`), late-day humidity (`Humidity3pm`), and wind gust velocity (`WindGustSpeed`) emerged as the strongest predictors, matching meteorological dynamics.

### 6. Deployment
- Built a unified `Pipeline` combining the `ColumnTransformer` preprocessor and the trained `LGBMClassifier`.
- Serialized the end-to-end model into `weather_rain_prediction_lgbm_pipeline.joblib`.
- Developed a production-ready `predict_rain_tomorrow()` inference function capable of processing raw data frames and outputting risk levels.

---

## Tech Stack
- **Language:** Python
- **Core Libraries:** Scikit-Learn, LightGBM, Pandas, NumPy, Matplotlib, Seaborn, Joblib
- **Methodology:** CRISP-DM

---

## Repository Structure
```text
├── weather_rainfall_prediction_crispdm.ipynb  # Full CRISP-DM notebook
├── .gitignore                                # Ignores raw datasets and binary models
└── README.md                                 # Technical documentation
```
---

## How to Run
- Clone this repository: 
  ```bash
  git clone https://github.com/RenatoSaldivia/australia-weather-data-analysis.git
  ```
- Download the dataset (weatherAUS.csv) from Kaggle - Australia Weather Data and place it in the project root directory.
- Open weather_rainfall_prediction_crispdm.ipynb in Google Colab or Jupyter Notebook and run all cells sequentially.