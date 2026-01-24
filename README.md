# Interpretation of Machine Learning Models

## Contributors

Daisy (Ying) Zhou, William Song, Yasaman Baher

## Project Description

Creating machine learning models often involves writing redundant code, particularly when tuning hyperparameters and comparing performance across different models. This project aims to reduce that redundancy by streamlining these repetitive steps, making the model development process more efficient and time-effective. To achieve this, our project focuses on building reusable functions that, given user input, automatically return the optimal hyperparameters, the best-performing model, its accuracy score, and a corresponding confusion matrix, all in a single, unified workflow.

## List of Functions

-   `param_tuning_summary` <br> Creates a summary of the hyperparameter tuning results and extract the best estimator. <br>
-   `model_cv_metric_compare` <br> Creates a dataframe of cross-validation metric results between models for comparisons. <br>
-   `model_evaluation_plotting` <br> Creates standard classification metrics, creates a confusion matrix as a table, and creates a confusion matrix display object for visualization. <br>

## Positioning in the Python Ecosystem

This package is designed to effectively sit within the existing Python machine learning ecosystem, specifically the scikit-learn library for model training, hyperparameter tuning, and evaluation. While scikit-learn is a powerful library on its own, our functions aim to reduce repeated and manual comparisons between multiple models, something that scikit-learn lacks. Other packages such as mlxtend and yellowbrick offer visualization utilities for users; however, they tend to focus more on the visual aspects of models rather than providing a unified workflow. Our package targets this gap by combining hyperparameter tuning, model comparison, metric reporting, and confusion matrix generation into reusable functions, improving reproducibility and efficiency during model development.

## Installation

#### Install From PyPI:

``` base
pip install model-auto-interpret
```

#### Development Installation:

``` base
git clone https://github.com/UBC-MDS/model_interpretation.git
cd model_interpretation
pip install -e .
```

#### To Include Dependencies:

``` base
pip install -e ".[tests]"
```

## Reproducibility

This project is fully reproducible given the steps below. All experiments can be rerun end-to-end.

## Run from Source (Development)

``` bash
# 1. Clone the repository
git clone https://github.com/UBC-MDS/model_interpretation.git
cd model_interpretation

# 2. Create and activate environment
conda env create -f environment.yml
conda activate model_interp

# 3. Install the package
# Exclude tests if not needed
pip install -e .
# Include tests if needed
pip install -e ".[tests]"

# 4. Run tests
# If tests were installed in step 3
# For more details, see tests/README.md, linked below
pytest
```

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

## License

The software code contained within this repository is licensed under the [MIT license](https://spdx.org/licenses/MIT.html). See the license file for more information.