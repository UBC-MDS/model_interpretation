from sklearn.metrics import ConfusionMatrixDisplay, fbeta_score
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

def model_evaluation_plotting(pipeline, X_test, y_test, display_labels=None):
    """
    Compute classification metrics and generate confusion matrix visualizations.

    Parameters
    ----------
    pipeline : sklearn estimator or sklearn.pipeline.Pipeline
        Fitted model object that implements .predict() and .score() methods.
    X_test : pandas.DataFrame
        Test feature matrix. Must not contain NaN values.
    y_test : array-like of shape (n_samples,)
        True labels for test data. Can be pandas Series, numpy array, or list.
        Must not contain NaN values.
    display_labels : array-like of shape (n_classes,), optional
        Target class labels for confusion matrix display. If None, uses numeric labels.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'accuracy' : float - Classification accuracy on test data
        - 'f2' : float - F2 score (beta=2)
        - 'y_pred' : numpy.ndarray - Predicted labels
    cm_table : pandas.DataFrame
        Confusion matrix as a crosstab (rows=true labels, columns=predicted labels).
    cm_display : sklearn.metrics.ConfusionMatrixDisplay
        Confusion matrix display object for visualization.

    Raises
    ------
    TypeError
        If pipeline is not fitted, or if input types are invalid.
    ValueError
        If X_test and y_test have different lengths, or contain NaN values.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> import pandas as pd
    >>> 
    >>> X, y = load_iris(return_X_y=True)
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> model = SVC().fit(X_train, y_train)
    >>> metrics, cm_table, cm_display = model_evaluation_plotting(
    ...     model, X_test, y_test, 
    ...     display_labels=['setosa', 'versicolor', 'virginica']
    ... )
    >>> 
    >>> print(metrics['accuracy'])
    0.9667
    >>> print(cm_table)
    Predicted    0   1   2
    Actual               
    0           10   0   0
    1            0   9   1
    2            0   0  10
    >>> cm_display.plot()  # Shows confusion matrix visualization

    Notes
    -----
    - The model must be fitted before calling this function
    - F2 score emphasizes recall over precision (beta=2)
    - For binary classification, uses pos_label="Y"
    """
    # Basic interface checks

    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "score"):
        raise TypeError("pipeline must have predict and score methods")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame")

    if y_test is None or not isinstance(y_test, (pd.Series, np.ndarray, list)):
        raise TypeError("y_test must be a 1D array-like (Series/ndarray/list), not None")

    # Convert y_test to Series for consistency
    y_test = pd.Series(y_test)

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length")
    
    # Ensure model is fitted
    try:
        check_is_fitted(pipeline)
    except Exception as e:
        raise TypeError("pipeline must be fitted before calling model_evaluation_plotting") from e

    # NaN checks
    if X_test.isna().any().any():
        raise ValueError("X_test contains NaNs")

    if y_test.isna().any():
        raise ValueError("y_test contains NaNs")

    # Compute predictions and metrics
    accuracy = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)

    f2 = fbeta_score(y_test, y_pred, beta=2, pos_label="Y")

    # Confusion matrix (table + visualization)
    cm_table = pd.crosstab(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, values_format="d", display_labels=display_labels
    )

    return accuracy, f2, y_pred, cm_table, cm_display