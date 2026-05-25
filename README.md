# Buy Now Pay Later Adoption and Delinquency Prediction using SHED data - a multi-model comparison

![](./img/cover.png)

## Overview
This analysis utilizes data from the Federal Reserve’s Survey of Household Economics and Decisionmaking (SHED) for the years 2022–2025 to model the adoption and delinquency patterns of Buy Now, Pay Later (BNPL) users.
**Target Variables:**
BNPL Adoption: Derived from question BNPL1 ("In the past year, have you used a “Buy Now, Pay Later” service to buy something?")
BNPL Delinquency: Derived from question BNPL3 ("In the past year, have you ever been late making a payment for something you bought using a Buy Now, Pay Later service?")

**Predictive Features:**
The model relies on self-reported features across several financial and demographic dimensions:
Financial Well-being: Economic status, financial concerns, emergency fund availability, and savings behaviors.
Credit & Payment Habits: General payment behavior and credit card utilization.
Socio-demographics: Race, educational attainment, total household assets, and number of children under 18."

## Data
The analysis uses data from the following sources:
https://www.federalreserve.gov/consumerscommunities/shed_data.htm
- **Dataset 1**: 2022 Survey Data csv file
- **Dataset 2**: 2023 Survey Data cvs file
- **Dataset 2**: 2024 Survey Data cvs file
- **Dataset 2**: 2025 Survey Data cvs file

Due to inconsistencies in survey IDs and answers across the four years, data processing was performed to synchronize and merge the datasets.

## Predictors:
- [Feature 1]
- [Feature 2]
- [Feature 3]

## Target variables:
- BNPL1
- BNPLE

## Methods
Dimensionality reduction: Due to the size of predictive variables, PCA and step-wise feature selection are performed for dimension reduction

Modeling: We apply several statistical learning models, including:
- Logistic regression
- SVM
- XGBoost
- Deep Learning

Performance evaluation: model performance is evaluated using:
- RMSE and R² (regression)
- Accuracy, Recall and ROC-AUC (classification)

## Results
Key findings include:
- **[Insight 1]**
- **[Insight 2]**
- **[Insight 3]**

These results suggest that **[interpretation]**.

## Repository Structure
```text
├── data/               # Raw and processed datasets
├── image/              # Plots and images
├── src/                # Code
├── README.md           # Project documentation
