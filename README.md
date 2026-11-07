# Credit Card Default Prediction Using Machine Learning and Explainable AI

## Project Overview

This project develops an end-to-end machine learning solution to predict whether a credit card customer is likely to default on their payment in the following month.

The project demonstrates a complete data science workflow, including data cleaning, exploratory data analysis (EDA), SQL analysis, feature engineering, machine learning, model evaluation, classification threshold analysis, probability calibration, SHAP explainability and deployment using Streamlit.

The aim is to explore how machine learning can support credit risk analysis while maintaining transparency and responsible model interpretation.

## Live Application

**Try the interactive credit card default prediction application:**

[Launch Credit Card Default Prediction App](https://creditcarddefaultprediction-4hk6sclth5dqycvkzddnsn.streamlit.app/)

The application allows users to enter customer financial and repayment information and receive a model-generated estimate of the probability of default in the following month.

**Disclaimer:** This is an educational portfolio project using historical Taiwanese credit card data. The model is not validated for real-world lending decisions or deployment in UK banking.

## Business Problem

Credit card default represents a financial risk for banks and other lending institutions.

Understanding which customers may be more likely to default can support credit risk analysis, portfolio monitoring and the development of risk assessment approaches.

This project addresses the following question:

**Can historical repayment behaviour and customer financial information be used to predict credit card default in the following month?**

### Project Objectives

- Understand and clean the credit card dataset.
- Explore customer characteristics and default patterns using Python and SQL.
- Engineer features that summarise repayment behaviour.
- Train and compare classification models.
- Evaluate model performance using appropriate metrics.
- Investigate the trade-off between precision and recall.
- Assess predicted probabilities through calibration analysis.
- Explain model predictions using SHAP.
- Deploy an interactive prediction application using Streamlit.

## Dataset

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

The dataset contains information about 30,000 credit card clients in Taiwan.

It includes demographic characteristics, credit limits, historical repayment statuses, billing amounts and previous payments.

### Dataset Summary

| Characteristic | Description |
|---|---|
| Observations | 30,000 |
| Prediction target | Default payment in the following month |
| Problem type | Binary classification |
| Default class | 1 |
| Non-default class | 0 |
| Default rate | Approximately 22% |
| Geographic context | Taiwan |
| Currency | New Taiwan dollars (NTD) |

### Important Variables

- `LIMIT_BAL`: Customer credit limit.
- `AGE`: Customer age.
- `SEX`, `EDUCATION`, `MARRIAGE`: Demographic characteristics.
- `PAY_1` to `PAY_6`: Historical repayment-status codes.
- `BILL_AMT1` to `BILL_AMT6`: Historical billing amounts.
- `PAY_AMT1` to `PAY_AMT6`: Historical payment amounts.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | pandas, NumPy |
| Database | SQLite, SQL |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Explainable AI | SHAP |
| Model persistence | joblib |
| Web application | Streamlit |
| Development | VS Code, Jupyter Notebook |
| Version control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## Project Workflow

The project follows these stages:

1. Data understanding and cleaning
2. Exploratory data analysis
3. SQL-based analysis
4. Feature engineering
5. Baseline model development
6. Model comparison and selection
7. Model evaluation
8. Classification threshold analysis
9. Probability calibration
10. SHAP explainability
11. Streamlit application development
12. GitHub publication and cloud deployment

## Exploratory Data Analysis

Exploratory data analysis was conducted to understand the dataset and investigate patterns associated with credit card default.

The analysis included:

- Distribution of default and non-default customers.
- Customer demographic characteristics.
- Credit limit distributions.
- Historical repayment behaviour.
- Billing and payment amount distributions.
- Relationships between customer characteristics and default outcomes.

The dataset is imbalanced, with approximately 22% of customers belonging to the default class.

This makes accuracy alone insufficient for evaluating the model.

## SQL Analysis

SQLite was used to perform structured queries and investigate patterns within the credit card dataset.

Four SQL queries were developed as part of the analysis.

The SQL component demonstrates how database querying can support exploratory analysis and business-oriented investigation.

SQL scripts are available in the `sql/` directory.

## Feature Engineering

Two repayment-history features were engineered to summarise customer payment behaviour.

### Months With Delay

`months_with_delay` counts the number of historical repayment periods with a positive repayment-delay status code.

This feature captures the frequency of recorded repayment difficulties.

### Maximum Delay

`max_delay` represents the highest positive repayment-delay status code across the six historical periods.

This feature summarises the most severe recorded repayment-status category.

These features were included alongside the original customer information during model development.

## Machine Learning Models

Two supervised classification algorithms were evaluated.

### Logistic Regression

Logistic Regression was used as the baseline classification model because of its simplicity and relative interpretability.

### Random Forest

Random Forest was evaluated as an alternative model capable of capturing nonlinear relationships and interactions between features.

The modelling workflow used scikit-learn pipelines to combine preprocessing and classification.

### Final Model Selection

The final model was selected following comparison of classification performance and consideration of the credit default prediction problem.

**Selected final model:** [Enter Logistic Regression or Random Forest, according to your notebook]

## Model Evaluation and Results

The final model was evaluated using a held-out test dataset containing **6,000 customers**.

Of these:

- **4,673 customers** did not default.
- **1,327 customers** defaulted.

The model was assessed using precision, recall, F1-score, accuracy, ROC-AUC and Average Precision.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-default (0) | 0.84 | 0.94 | 0.89 | 4,673 |
| Default (1) | 0.65 | 0.36 | 0.46 | 1,327 |
| Macro average | 0.74 | 0.65 | 0.68 | 6,000 |
| Weighted average | 0.80 | 0.82 | 0.79 | 6,000 |

### Overall Performance

| Metric | Test-set result |
|---|---:|
| Accuracy | 0.82 |
| ROC-AUC | 0.773 |
| Average Precision (PR-AUC summary) | 0.5509 |
| Precision — Default Class | 0.65 |
| Recall — Default Class | 0.36 |
| F1-score — Default Class | 0.46 |

### Interpretation of Results

**1. Overall Accuracy — 82%**

The model correctly classified approximately 82% of customers in the test dataset.

However, because the dataset contains substantially more non-default customers than default customers, accuracy alone does not provide a complete assessment of model performance.

**2. ROC-AUC — 0.773**

The ROC-AUC score indicates that the model has a reasonable ability to distinguish between customers who default and those who do not.

A score of 0.773 suggests useful predictive discrimination, although there is room for improvement.

**3. Average Precision — 0.5509**

Average Precision summarises the precision-recall relationship across classification thresholds.

This metric is particularly relevant because the default class represents a minority of customers.

The result indicates that the model has meaningful ranking ability for identifying potential defaults, but its predictions are not sufficiently reliable for real-world lending decisions without further validation.

**4. Precision for Default Customers — 65%**

Among customers predicted to default, approximately 65% actually belonged to the default class.

This indicates that the model's positive predictions have moderate precision.

**5. Recall for Default Customers — 36%**

The model identified approximately 36% of customers who actually defaulted.

This means that approximately 64% of actual default cases were not identified at the evaluated classification threshold.

This is an important limitation in a credit risk context, where failing to identify potential defaults may have financial consequences.

**6. F1-score for Default Customers — 0.46**

The default-class F1-score reflects the trade-off between precision and recall.

The relatively low score highlights the difficulty of identifying the minority default class.

### Key Finding

The model demonstrates useful discrimination, with an ROC-AUC of 0.773, but its default-class recall of 0.36 indicates that many actual default cases are missed at the evaluated threshold.

This illustrates why credit risk models should be assessed using class-specific metrics, threshold analysis and business considerations rather than accuracy alone.

## Classification Threshold Analysis

Classification threshold analysis was conducted to investigate the trade-off between precision and recall.

In credit risk modelling, different prediction errors can have different consequences:

- **False negatives:** Customers who default but are predicted not to default.
- **False positives:** Customers who do not default but are predicted to default.

Lowering the classification threshold may improve recall for the default class, but can also increase false positives and reduce precision.

The analysis demonstrates why threshold selection should consider business objectives, error costs and model performance.

## Probability Calibration

Probability calibration was investigated to assess how closely predicted default probabilities align with observed default frequencies.

This is relevant because a credit risk model may be used to estimate the likelihood of default rather than simply produce a binary prediction.

Calibration analysis complements classification metrics by examining the reliability of predicted probabilities.

## Model Explainability Using SHAP

SHAP (SHapley Additive exPlanations) was used to investigate how individual features contribute to the model's predictions.

The explainability analysis includes:

- **Global feature importance:** Identifies variables with the largest average absolute SHAP contributions.
- **SHAP beeswarm analysis:** Shows the distribution and direction of feature contributions across customers.
- **Individual waterfall explanation:** Demonstrates how customer features contribute to a particular model prediction.
- **Business interpretation:** Explores the relevance of model explanations to credit risk analysis.
- **Limitations:** Discusses correlated variables, causality, historical data and model governance.

Repayment-status variables, historical payment behaviour and credit limits are relevant features for interpreting the model's behaviour.

[Read the Detailed SHAP Explainability Report](reports/shap_explainability.md)

## Project Structure

```text
credit-card-default-prediction/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── models/
│   └── best_credit_risk_model.joblib
├── notebooks/
├── reports/
│   ├── figures/
│   └── shap_explainability.md
├── sql/
└── src/
```

## Running the Application Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-default-prediction.git
cd credit-card-default-prediction
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

For macOS or Linux:

```bash
source .venv/bin/activate
```

For Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

The application should open in your browser.

Local execution requires the saved model file and compatible Python and scikit-learn versions.

## Limitations

**1. Historical Dataset**

The dataset contains historical customer information and may not represent current credit card behaviour.

**2. Geographic Differences**

The dataset comes from Taiwan and may not generalise to UK banking customers.

**3. Class Imbalance**

Approximately 22% of customers belong to the default class, creating challenges for model evaluation and classification.

**4. Limited Default Recall**

The model identifies only 36% of actual default cases at the evaluated threshold.

This is a significant limitation for applications where identifying potential defaults is a priority.

**5. Model Explainability**

SHAP provides explanations of model behaviour but does not establish causal relationships.

**6. Fairness and Bias**

Demographic variables require careful assessment for potential bias and discriminatory outcomes.

**7. Production Readiness**

The application is an educational demonstration and has not undergone the validation, security, monitoring or regulatory assessment required for real-world lending decisions.

## Future Improvements

Potential improvements include:

- Investigating additional classification algorithms.
- Performing more extensive hyperparameter tuning.
- Optimising classification thresholds based on business error costs.
- Improving recall while monitoring precision.
- Evaluating probability calibration on additional data.
- Assessing fairness across customer subgroups.
- Investigating model stability and performance drift.
- Testing the approach using more recent and geographically relevant data.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for credit card default prediction.

It combines Python, SQL, exploratory analysis, feature engineering, classification modelling, evaluation, SHAP explainability and Streamlit deployment.

The selected model achieved a **ROC-AUC of 0.773**, **Average Precision of 0.5509** and **82% accuracy** on the held-out test dataset.

However, default-class recall was only **36%**, demonstrating the importance of class-specific evaluation and careful threshold selection.

The project highlights both the potential and limitations of applying explainable machine learning to credit risk analysis.

**This application is intended for educational and portfolio demonstration purposes only.**

## Live Application
Try the interactive credit card default prediction app: 
https://creditcarddefaultprediction-4hk6sclth5dqycvkzddnsn.streamlit.app/
