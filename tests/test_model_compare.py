import pytest
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from model_auto_interpret.utils import create_test_artifacts 
from model_auto_interpret.model_compare import model_cv_metric_compare


@pytest.fixture
def shared_artifacts():
    """
    Pytest fixture to generate synthetic data and models once for all tests.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, models) where models is a dict of fitted pipelines.
    """
    return create_test_artifacts()

def test_input_validation_errors(shared_artifacts):
    """
    Test that the function raises correct errors for invalid inputs.
    """
    X_train, _, y_train, _, models = shared_artifacts

    # Test models_dict type check
    # Pass a list instead of a dict
    with pytest.raises(TypeError, match="must be a dictionary"):
        model_cv_metric_compare(["not_a_dict"], X_train, y_train)

    # Test empty models_dict check
    # Pass an empty dict
    with pytest.raises(ValueError, match="cannot be empty"):
        model_cv_metric_compare({}, X_train, y_train)

    # Test X type check
    # Pass a numpy array instead of a DataFrame
    with pytest.raises(TypeError, match="'X' must be a pandas DataFrame"):
        model_cv_metric_compare(models, X_train.values, y_train)

    # Test y type check
    # Pass a numpy array instead of a Series
    with pytest.raises(TypeError, match="'y' must be a pandas Series"):
        model_cv_metric_compare(models, X_train, y_train.values)

    # Test y content check
    # Create a target series that only has 'N', missing the required 'Y'
    y_invalid = pd.Series(['N'] * len(y_train))
    with pytest.raises(ValueError, match="must contain the class label 'Y'"):
        model_cv_metric_compare(models, X_train, y_invalid)

def test_cv_returns_dataframe(shared_artifacts):
    """
    Test if the function returns a non-empty Pandas DataFrame with the correct index.
    
    This ensures the basic output structure is correct before checking specific values.
    """
    X_train, _, y_train, _, models = shared_artifacts

    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.index.name == "Model"

def test_cv_includes_all_metrics(shared_artifacts):
    """
    Test if the output DataFrame contains all required metric columns.
    
    Verifies that 'accuracy', 'precision', 'recall', 'f1', and 'roc_auc' are present
    in the results.
    """
    X_train, _, y_train, _, models = shared_artifacts
    
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    expected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in expected_metrics:
        assert metric in result.columns

def test_roc_auc_logic(shared_artifacts):
    """
    Test if ROC AUC is correctly calculated for models that support probability.
    
    Verifies that a model with `predict_proba` (like RandomForest) receives a 
    valid float score for ROC AUC, rather than NaN.
    """
    X_train, _, y_train, _, models = shared_artifacts
    
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    # Check Random Forest (it supports probabilities)
    assert "roc_auc" in result.columns
    assert not np.isnan(result.loc["RandomForest", "roc_auc"])

def test_values_are_means(shared_artifacts):
    """
    Test if the returned metric values are valid floating point numbers (0.0 to 1.0).
    
    A sanity check to ensure the cross-validation mean calculation 
    is working and not returning strings or unscaled values.
    """
    X_train, _, y_train, _, models = shared_artifacts
    
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    acc = result.loc["RandomForest", "accuracy"]
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_mixed_probability_support(shared_artifacts):
    """
    Test robust handling of mixed model capabilities (with and without `predict_proba`).
    
    Ensures that when comparing a model that supports probabilities (RF) against
    one that does not (SVM with probability=False), the function:
    1. Still produces an 'roc_auc' column.
    2. Assigns a valid score to the probability model.
    3. Assigns NaN correctly to the non-probability model.
    """
    X_train, _, y_train, _, models = shared_artifacts
    
    # Create a model explicitly without probability support
    svm_no_prob = make_pipeline(StandardScaler(), SVC(probability=False))
    
    mixed_models = {
        "RF": models["RandomForest"], 
        "SVM_NoProb": svm_no_prob
    }

    # Runs the code path where `if hasattr(...)` is False
    result = model_cv_metric_compare(mixed_models, X_train, y_train, cv=2)
    
    # Assertions to ensure it handled it correctly
    assert np.isnan(result.loc["SVM_NoProb", "roc_auc"])