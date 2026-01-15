import pytest
import pandas as pd
import numpy as np
from src.model_evaluation import model_evaluation_plotting
from src.utils import create_test_artifacts
from sklearn.metrics import ConfusionMatrixDisplay

@pytest.fixture(scope="module")
def artifacts():
    return create_test_artifacts()

@pytest.fixture
def X_train(artifacts):
    return artifacts[0]

@pytest.fixture
def y_train(artifacts):
    return artifacts[2]

@pytest.fixture
def pipeline(artifacts):
    models = artifacts[4]
    return models["Dummy"]

def test_model_evaluation_plotting(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)

    accuracy, f2, y_pred, cm_table, cm_display = model_evaluation_plotting(
    pipeline, X_train, y_train)

    assert "Y" in set(y_train), "y_train must contain 'Y' because pos_label='Y' is hardcoded"
    assert isinstance(accuracy, (float, np.floating))
    assert 0.0 <= float(accuracy) <= 1.0

    assert isinstance(f2, (float, np.floating))
    assert 0.0 <= float(f2) <= 1.0

    assert isinstance(y_pred, (np.ndarray, list, pd.Series))
    assert len(y_pred) == len(y_train)

    assert isinstance(cm_table, pd.DataFrame)
    assert cm_table.shape[0] >= 1 and cm_table.shape[1] >= 1


def test_model_evaluation_plotting_wrong_input(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)

    with pytest.raises(TypeError):
        model_evaluation_plotting(123, X_train, y_train)

    with pytest.raises(TypeError):
        model_evaluation_plotting(pipeline, "not a dataframe", y_train)

    with pytest.raises(TypeError):
        model_evaluation_plotting(pipeline, X_train, None)

def test_model_evaluation_plotting_return_types(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)

    accuracy, f2, y_pred, cm_table, cm_display = model_evaluation_plotting(
        pipeline, X_train, y_train
    )

    assert isinstance(accuracy, (float, np.floating))
    assert isinstance(f2, (float, np.floating))
    assert isinstance(cm_table, pd.DataFrame)
    assert isinstance(cm_display, ConfusionMatrixDisplay)