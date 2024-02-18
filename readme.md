# Predictive Analytics for Employee Attrition
This project aims to model the probability of employee attrition and provide actionable insights for a company. The predictive model helps identify key factors contributing to attrition and offers recommendations for addressing potential issues. The project includes exploratory data analysis, pre-processing, model development, and a proposed machine learning pipeline for future use.

### Dataset
The dataset encapsulates essential aspects of employee engagement and work dynamics. It encompasses key parameters like satisfaction levels, project involvement, and work hours, offering a succinct yet insightful snapshot of the workforce. This dataset provides valuable information to discern patterns, identify influential factors, and derive actionable insights to enhance employee satisfaction and retention strategies.

### Model Selection
The choice of XGBoost (eXtreme Gradient Boosting) is grounded in its efficiency and scalability, especially for large datasets and complex models. It stands out for its capacity to manage missing values, categorical variables, and high-dimensional data, making it aptly suited for the project. Moreover, it encompasses regularization techniques that deter overfitting and enhance model generalization. 

### Model Architecture
The project employs a gradient boosting tree ensemble model within the XGBoost framework. This model synergizes predictions from several decision trees to arrive at a final prediction for the target variable. A set of key parameters, including learning rate, maximum tree depth, and number of trees in the ensemble, govern the model, allowing for fine-tuning to attain optimal performance on the dataset.

The model uses the following key parameters:

- `learning_rate`: The step size of the optimization algorithm
- `max_depth`: The maximum depth of a tree
- `n_estimators`: The number of of trees in the ensemble
- `gamma`: Minimum split loss
- `lambda`: Regularization term
- `scale_pos_weight`: The balance between positive and negative weights
- `min_child_weight`: Minimum sum of weights of all observations in a child
- `objective`: Loss function
- `tree_method`: Method used to grow the tree

By adjusting these parameters, the model can be fine-tuned to achieve the best performance on the given dataset.

### Model Performance
With meticulous tuning and optimization, the model has achieved a promising accuracy rate of `99.23%` on the test dataset.

### Requirements
Execute `pip install -r requirements.txt` to install the required libraries.

### Exectuion
Execute `employeeattrition.sh` to initiate the entire pipeline.

### Unit Test
Execute `python -m unittest discover test` to run all unit tests, ensuring the reliability of the code base.

### Developer
Execute `python -m pre_commit run --all-files` to ensure code quality and formatting checks.
