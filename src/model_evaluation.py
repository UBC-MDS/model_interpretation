from sklearn.metrics import ConfusionMatrixDisplay, fbeta_score
import pandas as pd
import numpy as np

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
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "score"):
        raise TypeError("pipeline must have predict and score methods")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame")

    if not isinstance(y_test, (pd.Series, np.ndarray, list)):
        raise TypeError("y_test must be a 1D array-like")
    
    accuracy = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)

    f2 = fbeta_score(y_test, y_pred, beta=2, pos_label="Y")
    cm_table = pd.crosstab(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, values_format="d")

    return accuracy, f2, y_pred, cm_table, cm_display

