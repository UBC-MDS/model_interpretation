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
    Calls the helper function ONCE.
    This creates the data and models used by all tests.
    """
    return create_test_artifacts()

def test_cv_returns_dataframe(shared_artifacts):
    # Unpack what you need from the fixture
    X_train, X_test, y_train, y_test, models = shared_artifacts
    
    # Use X_train/y_train for Cross Validation
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.index.name == "Model"

def test_cv_includes_all_metrics(shared_artifacts):
    X_train, X_test, y_train, y_test, models = shared_artifacts
    
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    expected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in expected_metrics:
        assert metric in result.columns

def test_roc_auc_logic(shared_artifacts):
    X_train, X_test, y_train, y_test, models = shared_artifacts
    
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    # Check Random Forest (it supports probabilities)
    assert "roc_auc" in result.columns
    assert not np.isnan(result.loc["RandomForest", "roc_auc"])

def test_values_are_means(shared_artifacts):
    X_train, X_test, y_train, y_test, models = shared_artifacts
    
    result = model_cv_metric_compare(models, X_train, y_train, cv=2)
    
    acc = result.loc["RandomForest", "accuracy"]
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_mixed_probability_support(shared_artifacts):
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