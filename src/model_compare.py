import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def model_metric_compare(pipeline, X_test, y_test, model_name):
    """
    Create dataframe of metric results between models for comparisons.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline object.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series
        Test labels.
    model_name : str
        Name of the model.

    Returns
    -------
    dataframe : pandas.DataFrame
        Dataframe containing model name and evaluation metrics.
    """
    
    return None