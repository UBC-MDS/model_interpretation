# Interpretation of Machine Learning Models

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Package Version](https://img.shields.io/badge/version-0.0.4-green.svg)](https://github.com/UBC-MDS/model_interpretation) [![Build Status](https://github.com/UBC-MDS/model_interpretation/actions/workflows/build.yml/badge.svg)](https://github.com/UBC-MDS/model_interpretation/actions) [![codecov](https://codecov.io/gh/UBC-MDS/model_interpretation/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/model_interpretation) [![PyPI version](https://img.shields.io/pypi/v/model-auto-interpret)](https://pypi.org/project/model-auto-interpret/)

## Project Description

Creating machine learning models often involves writing redundant code, particularly when tuning hyperparameters and comparing performance across different models. This project aims to reduce that redundancy by streamlining these repetitive steps, making the model development process more efficient and time-effective. To achieve this, our project focuses on building reusable functions that, given user input, automatically return the optimal hyperparameters, the best-performing model, its accuracy score, and a corresponding confusion matrix, all in a single, unified workflow.

## List of Functions

-   `param_tuning_summary` <br>

    **Purpose:**<br>\
    Summarizes the results of a hyperparameter search and gets the best-performing model.<br>

    **Detailed explanation:**<br>\
    This function takes a fitted `GridSearchCV` or `RandomizedSearchCV` object and converts the cross-validation results into a readable dataframe. It shows how different hyperparameter combinations performed when tuning and returns the estimator that had the best cross-validation score. This allows users to inspect tuning results and proceed with the best model without manually accessing `cv_results_` or `best_estimator_`.<br>

    **Typical use case:**<br>\
    Use this function after hyperparameter tuning to review model performance across parameter combinations and get the best model for evaluation or deployment<br>

-   `model_cv_metric_compare` <br>

    **Purpose:**<br>\
    Compares multiple machine learning models using cross-validation metrics.<br>

    **Detailed explanation:**<br>\
    This function takes a dictionary of models that have not been trained and evaluates each one using cross-validation on the same dataset. It computes performance metrics across models and returns a dataframe summarizing their average cross-validation performance. This helps users compare different performance under the same setup before selecting a final model.<br>

    **Typical use case:**<br>\
    Use this function when deciding which model (e.g., logistic regression vs. random forest vs. SVM) performs best for a given classification task.<br>

-   `model_evaluation_plotting` <br>

    **Purpose:**<br>\
    Evaluates a trained classification model on test data and visualizes its performance.<br>

    **Detailed explanation:**<br>\
    This function computes the standard classification metrics on test data, creates a confusion matrix in tabular form, and creates a confusion matrix visualization object. It gives both numerical and visual insights into model performance, making it easier to diagnose misclassifications and class-specific errors.<br>

    **Typical use case:**<br>\
    Use this function after training and selecting a final model to assess the performance of a model and visualize its prediction errors.<br>

## Positioning in the Python Ecosystem

This package is designed to effectively sit within the existing Python machine learning ecosystem, specifically the scikit-learn library for model training, hyperparameter tuning, and evaluation. While scikit-learn is a powerful library on its own, our functions aim to reduce repeated and manual comparisons between multiple models, something that scikit-learn lacks. Other packages such as mlxtend and yellowbrick offer visualization utilities for users; however, they tend to focus more on the visual aspects of models rather than providing a unified workflow. Our package targets this gap by combining hyperparameter tuning, model comparison, metric reporting, and confusion matrix generation into reusable functions, improving reproducibility and efficiency during model development.

## Installation

### Prerequisites

-   Python 3.10 or higher
-   pip package manager
-   (Optional) conda for environment management

### Option 1: Install from PyPI (Recommended for Users)

Install the latest stable version:

``` bash
pip install -i https://test.pypi.org/simple/model-auto-interpret
```

### Option 2: Install from Source (For Development)

For contributors or those who want the latest development version:

``` bash
# 1. Clone the repository
git clone https://github.com/UBC-MDS/model_interpretation.git
cd model_interpretation

# 2. Create and activate a conda environment (recommended)
conda env create -f environment.yml
conda activate model_interp

# Alternative: Use venv instead of conda
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the package in editable mode
pip install -e .

# 4. (Optional) Install development dependencies
pip install -e ".[dev]"      # For linting and formatting
pip install -e ".[tests]"    # For running tests  
pip install -e ".[docs]"     # For building documentation

# 5. Run tests to verify installation
pytest
```

## Quick Start

Here's a complete example showing all three main functions:

``` python
from model_auto_interpret import (
    param_tuning_summary,
    model_cv_metric_compare,
    model_evaluation_plotting
)
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load and prepare data
X, y = load_iris(return_X_y=True)
X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = pd.Series(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Example 1: Hyperparameter Tuning Summary
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

summary_df, best_model = param_tuning_summary(grid_search)
print("Best Parameters:")
print(summary_df)

# Example 2: Compare Multiple Models  
models = {
    'SVC': SVC(),
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

comparison_df = model_cv_metric_compare(models, X_train, y_train, cv=5)
print("\nModel Comparison:")
print(comparison_df)

# Example 3: Evaluate and Visualize
metrics, cm_table, cm_display = model_evaluation_plotting(
    best_model,
    X_test,
    y_test,
    display_labels=['setosa', 'versicolor', 'virginica']
)

print("\nEvaluation Metrics:")
print(metrics)

# Display confusion matrix
cm_display.plot()
plt.show()
```

**For more detailed examples and tutorials, visit our [documentation website](https://machine-learning-model-interpretation.netlify.app/).**

### Verify Installation

After installation, verify everything works:

``` python
import model_auto_interpret
print(f"Version: {model_auto_interpret.__version__}")
```

Expected output: `Version: 0.1.2` (or current version)

## Running the test suite

Tests are run using the `pytest` command in the root of the project. More details about the test suite and to run tests can be found in the [`tests`](tests) directory.

## Environment and Dependencies

``` yaml
name: model_interp
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.4.1
  - pandas=2.3.3
  - scikit-learn
  - jupyterlab
  - pytest=9.0.2          
  - pytest-cov=7.0.0
  - matplotlib=3.10.8
```

To recreate this environment exactly:

``` bash
conda env create -f environment.yml
```

To update dependencies:

``` bash
conda env update -f environment.yml --prune
```

## Contributors

Daisy (Ying) Zhou, William Song, Yasaman Baher

## License

The software code contained within this repository is licensed under the [MIT license](https://spdx.org/licenses/MIT.html). See the license file for more information.
