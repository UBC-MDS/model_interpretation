import sys
from pathlib import Path
import pandas as pd
import pytest
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

@pytest.fixture
def sample_data():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = pd.Series([0, 1, 0, 1])
    return X, y

@pytest.fixture
def preprocessor():
    return make_column_transformer(
        (StandardScaler(), ["a", "b"]),
        remainder="drop",
    )