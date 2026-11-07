# Exploratory Data Analysis — Credit Card Default Prediction

## 1. Business Objective
Investigate historical customer credit behaviour and identify factors associated with next-month credit card default.

## 2. Dataset
- Source: UCI Default of Credit Card Clients
- Country: Taiwan
- Records: 30,000 in the original dataset
- Target: Next-month default indicator
- Currency: New Taiwan dollar (NTD)

## 3. Data Quality
- Missing values: 0
- Duplicate records: 0
- Unusual categorical values: [Insert findings]
- Cleaning decisions: [Explain changes]

## 4. Key Findings

### Finding 1 — Class Distribution
[Insert default rate and explain implications for modelling.]

### Finding 2 — Credit Limit
[Describe observed patterns and sample sizes.]

### Finding 3 — Repayment History
[Explain how repayment delay relates to default.]

### Finding 4 — Billing and Payment Behaviour
[Describe any associations and important limitations.]

## 5. Modelling Recommendations
- Establish a simple baseline.
- Investigate repayment-history features.
- Compare Logistic Regression and Random Forest.
- Use stratified splitting and training-only preprocessing.
- Evaluate ROC-AUC, PR-AUC, recall and calibration.

## 6. Limitations
The dataset represents historical Taiwanese credit card customers. Results cannot be assumed to generalise to modern UK banking customers. Observed associations do not establish causation.

## 7. Conclusion
[Summarise the most important evidence-supported findings and the next modelling steps.]