# Interpretation of Machine Learning Models

## Contributors
Daisy (Ying) Zhou, William Song, Yasaman Baher

## Project Description
Creating machine learning models often involves writing redundant code, particularly when tuning hyperparameters and comparing performance across different models. This project aims to reduce that redundancy by streamlining these repetitive steps, making the model development process more efficient and time-effective. To achieve this, our project focuses on building reusable functions that, given user input, automatically return the optimal hyperparameters, the best-performing model, its accuracy score, and a corresponding confusion matrix, all in a single, unified workflow.

## List of Functions
- ```param_tuning_summary``` <br>
Creates a summary of the hyperparameter tuning results and extract the best estimator. <br>
- ```model_metric_compare``` <br>
Creates a dataframe of metric results between models for comparisons.  <br>
- ```model_evaluation_plotting``` <br>
Creates standard classification metrics, creates a confusion matrix as a table, and creates a confusion matrix display object for visualization. <br>

## Positioning in the Python Ecosystem
This package is designed to effectively sit within the existing Python machine learning ecosystem, specifically the scikit-learn library for model training, hyperparameter tuning, and evaluation. While scikit-learn is a powerful library on its own, our functions aim to reduce repeated and manual comparisons between multiple models, something that scikit-learn lacks. Other packages such as mlxtend and yellowbrick offer visualization utilities for users; however, they tend to focus more on the visual aspects of models rather than providing a unified workflow. Our package targets this gap by combining hyperparameter tuning, model comparison, metric reporting, and confusion matrix generation into reusable functions, improving reproducibility and efficiency during model development.

## License
The software code contained within this repository is licensed under the [MIT license](https://spdx.org/licenses/MIT.html). See the license file for more information.