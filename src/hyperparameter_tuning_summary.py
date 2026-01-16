import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def param_tuning_summary(param_search_cv):
    """
    Create a summary of the hyperparameter tuning results and extract the best estimator

    Parameters
    ----------
    param_search_cv: a GridSearchCV or a RandomizedSearchCV instance

    Returns
    -------
    df_summary: pandas.DataFrame
        a Dataframe containing the best set of parameters' names, values and scores.
    
    best_estimator: the best model with the best set of parameters
    """
    # Validate input type
    if not isinstance(param_search_cv, (GridSearchCV, RandomizedSearchCV)):
        raise TypeError(
            "TypeError: input must be a GridSearchCV or RandomizedSearchCV instance."
        )
    
    # Check if the search has been fitted
    if not hasattr(param_search_cv, 'best_params_'):
        raise AttributeError(
            "AttributeError: The GridSearchCV/RandomizedSearchCV object has not been fitted yet. "
            "Please call .fit() before using param_tuning_summary()."
        )
    
    # Extract values needed
    best_score = param_search_cv.best_score_
    best_params = param_search_cv.best_params_
    best_estimator = param_search_cv.best_estimator_
    
    # summarize the best parameters with the vaue and scores
    data = {
        'Parameter': list(best_params.keys()),
        'Value': list(best_params.values()),
        'Best_Score': [best_score] * len(best_params)
    }
    df_summary = pd.DataFrame(data)
    
    return df_summary, best_estimator

    
