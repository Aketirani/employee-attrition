# Predictive Analytics For Employee Attrition
This project aims to model the probability of employee attrition and provide actionable insights for a company. The predictive model helps identify key factors contributing to attrition and offers recommendations for addressing potential issues. The project includes exploratory data analysis, pre-processing, model development, and a proposed machine learning pipeline for future use.

### Table of Contents
- [Structure](#structure)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Model Performance](#model-performance)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Execution](#execution)
- [Unit Test](#unit-test)
- [Developer](#developer)

### Structure
```
┌── config                      <-- Configuration Folder
|   └── *.yaml                  <-- Configuration Files
|
├── data                        <-- Data Folder
|   └── *.csv                   <-- Data Files
|
|
├── logs                        <-- Log Folder
|   └── *.log                   <-- Log Files
|
├── parameters                  <-- Parameters Folder
|   └── *.yaml                  <-- Model Parameters
|
├── plots                       <-- Plots Folder
|   └── *.png                   <-- Plots
|
├── results                     <-- Results Folder
|   └── *.yaml                  <-- Model Results
|
├── test                        <-- Test Folder
|   └── *.py                    <-- Unit Tests
|
├── .gitignore                  <-- Git Ignore Configuration
|
├── .pre-commit-config.yaml     <-- Pre-Commit Configuration
|
├── employee_attrition.py       <-- Main Python Script
|
├── employee_attrition.sh       <-- Main Shell Script
|
├── flowchart.wsd               <-- Pipeline Flowchart
|
├── readme.md                   <-- You Are Here
|
└── requirements.txt            <-- Package Requirements
```

### Dataset
The dataset encapsulates essential aspects of employee engagement and work dynamics. It encompasses key parameters like satisfaction levels, project involvement, and work hours, offering a succinct yet insightful snapshot of the workforce. This dataset provides valuable information to discern patterns, identify influential factors, and derive actionable insights to enhance employee satisfaction and retention strategies.

### Model Selection
The project utilizes two primary models: XGBoost (eXtreme Gradient Boosting) and Logistic Regression.

Logistic Regression is a statistical method used for binary classification, estimating the probability of an event occurrence based on one or more independent variables. It fits a logistic curve to the data to predict the likelihood of a binary outcome.

XGBoost (eXtreme Gradient Boosting) is a powerful gradient boosting framework that builds an ensemble of decision trees sequentially. It enhances traditional gradient boosting by incorporating regularization techniques and optimizing computational efficiency, leading to high predictive accuracy in classification and regression tasks.

### Model Performance
The Logistic Regression model has achieved an accuracy of `87.61%`, while the XGBoost model has achieved a promising accuracy of `99.15%`.

### Conclusion
In conclusion, XGBoost outperforms Logistic Regression in predicting employee attrition. Its superior performance highlights its effectiveness in capturing complex patterns within the data, making it the preferred choice for this project.

### Requirements
Execute `pip install -r requirements.txt` to install the required libraries.

### Execution
Execute `employee-attrition.sh -c config_file.yaml -m model_type` to initiate the entire pipeline.

Following arguments can be specified:
| Argument             | Description                    |
|----------------------|--------------------------------|
| `-c`, `--cfg_file`   | Path to the configuration file |
| `-m`, `--model_type` | Type of model to run           |

### Unit Test
Execute `python -m unittest discover test` to run all unit tests, ensuring the reliability of the code base.

### Developer
Execute `python -m pre_commit run --all-files` to ensure code quality and formatting checks.
