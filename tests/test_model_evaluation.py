import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

from src.model_evaluation import model_evaluation_plotting


@pytest.fixture
def dummy_pipeline():
    return make_pipeline(DummyClassifier())


@pytest.fixture
def dummy_X():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


@pytest.fixture
def dummy_y():
    return pd.Series(["N", "Y"])


def test_model_evaluation_plotting(dummy_pipeline, dummy_X, dummy_y):
    dummy_pipeline.fit(dummy_X, dummy_y)

    accuracy, f2, y_pred, cm_table, cm_display = model_evaluation_plotting(
    dummy_pipeline, dummy_X, dummy_y)

    assert isinstance(accuracy, (float, np.floating))
    assert 0.0 <= float(accuracy) <= 1.0

    assert isinstance(f2, (float, np.floating))
    assert 0.0 <= float(f2) <= 1.0

    assert isinstance(y_pred, (np.ndarray, list, pd.Series))
    assert len(y_pred) == len(dummy_y)

    assert isinstance(cm_table, pd.DataFrame)
    assert cm_table.shape[0] >= 1 and cm_table.shape[1] >= 1


def test_model_evaluation_plotting_wrong_input(dummy_pipeline, dummy_X, dummy_y):
    dummy_pipeline.fit(dummy_X, dummy_y)

    with pytest.raises(TypeError):
        model_evaluation_plotting(123, dummy_X, dummy_y)

    with pytest.raises(TypeError):
        model_evaluation_plotting(dummy_pipeline, "not a dataframe", dummy_y)

    with pytest.raises(TypeError):
        model_evaluation_plotting(dummy_pipeline, dummy_X, None)