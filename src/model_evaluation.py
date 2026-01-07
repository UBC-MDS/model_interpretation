import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def model_evaluation_plotting(pipeline, X_test, y_test):
    """
    Creates standard classification metrics, creates a
    confusion matrix as a table, and creates a confusion matrix
    display object for visualization.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline object.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series
        Test labels.

    Returns
    -------
    accuracy : float
        Classification accuracy on the test data.

    f2_score : float
        F2 score (beta = 2), emphasizing recall, computed with positive label "Y".

    y_pred : numpy.ndarray
        Predicted labels for the test data.

    cm_table : pandas.DataFrame
        Confusion matrix table with true labels as rows and predicted labels
        as columns.

    cm_display : sklearn.metrics.ConfusionMatrixDisplay
        Confusion matrix display object that can be plotted
    """
    return None