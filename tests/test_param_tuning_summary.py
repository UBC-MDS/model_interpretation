"""
A test module placeholder for param_tuning_summary function.
"""
import pytest
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_auto_interpret.hyperparameter_tuning_summary import param_tuning_summary
from model_auto_interpret.utils import create_test_search_cv_artifacts

@pytest.fixture(scope="module")
def search_artifacts():
    return create_test_search_cv_artifacts()

@pytest.fixture
def X_train(search_artifacts):
    return search_artifacts[0]

@pytest.fixture
def y_train(search_artifacts):
    return search_artifacts[2]

@pytest.fixture
def fitted_grid_search(search_artifacts):
    return search_artifacts[4]

@pytest.fixture
def fitted_random_search(search_artifacts):
    return search_artifacts[5]


def test_param_tuning_summary_unfitted_gridsearch():
    """
    Raise an AttributeError when given an unfitted GridSearchCV object. 
    Calling the function on an unfitted object 
    will fail to yield the best estimator from the instance.
    """
    # Create unfitted GridSearchCV
    param_grid = {'svc__C': [0.1, 1]}
    unfitted_gs = GridSearchCV(
        make_pipeline(StandardScaler(), SVC()),
        param_grid,
        cv=3
    )
    
    # Should raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        param_tuning_summary(unfitted_gs)
    
    # Verify error message is informative
    assert "not been fitted" in str(excinfo.value).lower()


def test_param_tuning_summary_invalid_input_type():
    """
    Raise a TypeError when given something that is not a GridSearchCV 
    or RandomizedSearchCV object. A wrong input type will fail to call 
    the function.
    """
    # Test with an estimator
    with pytest.raises(TypeError):
        param_tuning_summary(SVC())
        
    # Test with string
    with pytest.raises(TypeError):
        param_tuning_summary("not a search object")
    
    # Test with None
    with pytest.raises(TypeError):
        param_tuning_summary(None)
    
    # Test with list
    with pytest.raises(TypeError):
        param_tuning_summary([1,2,3])


def test_param_tuning_summary_single_parameter(X_train, y_train):
    """
    Make sure the tuning function correctly handles the edge case where
    only a single hyperparameter is being tuned.
    """
    # sample grid with only one hyperparameter
    param_grid = {'svc__C': [0.1, 1, 10]}
    
    grid_search = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(random_state=123)),
        param_grid,
        cv=3
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get summary
    df_summary, best_estimator = param_tuning_summary(grid_search)
    
    assert len(df_summary) == 1, "Should have 1 row for 1 parameter"
    assert df_summary['Parameter'].iloc[0] == 'svc__C'
    assert df_summary['Value'].iloc[0] in [0.1, 1, 10]
    assert isinstance(df_summary['Best_Score'].iloc[0], float)
    
    # verify the estimator makes correct number of predictions
    predictions = best_estimator.predict(X_train)
    assert len(predictions) == len(y_train)


def test_param_tuning_summary_pipeline_nested_params(fitted_grid_search, X_train, y_train):
    """
    Make sure the parameter names are nested (e.g., 'svc__C' instead of 'C') in the grids. 
    """
    df_summary, best_estimator = param_tuning_summary(fitted_grid_search)
    
    # validate if parameter names are nested and preserved
    param_names = df_summary['Parameter'].values
    assert 'svc__C' in param_names, "Hyperparameter name should be nested and preserved"
    assert 'svc__kernel' in param_names, "yperparameter name should be nested and preserved"
    
    # validate if there is a correct number of parameters
    assert len(df_summary) == 2, "Should have 2 parameters"

    
def test_param_tuning_summary_randomized_search(fitted_random_search, X_train, y_train):
    """
    Make sure the function also works with RandomizedSearchCV.
    """
    df_summary, best_estimator = param_tuning_summary(fitted_random_search)
    
    # test the output type and structure
    assert isinstance(df_summary, pd.DataFrame)
    assert list(df_summary.columns) == ['Parameter', 'Value', 'Best_Score']
    
    # test all params are included
    best_params = fitted_random_search.best_params_
    param_names = df_summary['Parameter'].values
    for param in best_params.keys():
        assert param in param_names, f"Parameter {param} is missing from the DataFrame"
    
    # test estimator
    assert hasattr(best_estimator, 'predict')
    predictions = best_estimator.predict(X_train)
    assert len(predictions) == len(y_train)


def test_param_tuning_summary_dataframe_structure(fitted_grid_search):
    """
    Tests that the output DataFrame has the structure expected.
    """
    df_summary, _ = param_tuning_summary(fitted_grid_search)
    
    # test column names
    assert list(df_summary.columns) == ['Parameter', 'Value', 'Best_Score'], "The output DataFrame must have columns: Parameter, Value, Best_Score"
    
    # tets if there are nulls
    assert not df_summary.isnull().any().any(), "The output DataFrame should have no nulls"
    
    # test if the scores are numeric and consistent
    assert pd.api.types.is_numeric_dtype(df_summary['Best_Score']), "Best_Score should be numeric"
    
    unique_scores = df_summary['Best_Score'].unique()
    assert len(unique_scores) == 1, "All rows should have the same Best_Score"
    
    # test if score is in [0, 1] 
    assert 0.0 <= unique_scores[0] <= 1.0, "The Best_Score should be between 0 and 1"


def test_param_tuning_summary_estimator_is_fitted(fitted_grid_search, X_train, y_train):
    """
    Tests the returned estimator is fitted and can make predictions.
    """
    _, best_estimator = param_tuning_summary(fitted_grid_search)
    predictions = best_estimator.predict(X_train)
    
    assert len(predictions) == len(y_train)
    if hasattr(best_estimator, 'predict_proba'):
        probas = best_estimator.predict_proba(X_train)
        assert probas.shape[0] == len(y_train)
        assert probas.shape[1] == 2, "Should have 2 columns for binary classification"


def test_param_tuning_summary_gridsearch_vs_randomized(fitted_grid_search, fitted_random_search):
    """
    Tests that both search work correctly and produce the same DataFrame structure
    """
    # get both df_summary from two searches
    grid_df, grid_estimator = param_tuning_summary(fitted_grid_search)
    random_df, random_estimator = param_tuning_summary(fitted_random_search)
    
    assert list(grid_df.columns) == list(random_df.columns), "GridSearchCV and RandomizedSearchCV should produce the structure of df_summary"
    assert hasattr(grid_estimator, 'predict')
    assert hasattr(random_estimator, 'predict')
    assert len(grid_df) == len(random_df) == 2, \
        "Both should have 2 parameters (svc__C and svc__kernel)"
    
# additional test functions added 

def test_param_tuning_summary_multiple_parameters(X_train, y_train):
    """
    Test that make sure the function handles GridSearchCV with multiple hyperparameters. 
    """

    # create a grid with over 2 parameters
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }
    
    # create a grid_search instance with the param_grid created
    grid_search = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(random_state=123)),
        param_grid,
        cv=3
    )

    grid_search.fit(X_train, y_train)
    
    df_summary, best_estimator = param_tuning_summary(grid_search)
    
    # check if the summary has 3 rows 
    assert len(df_summary) == 3, "The output Dtaframe should have 3 rows as there are 3 parameters"

def test_param_tuning_summary_parameter_value_types(fitted_grid_search):
    """
    Make sure the function preserves different parameter data types:
    The 'Value' column should maintain the original data types from the best parameters.
    """
    df_summary, _ = param_tuning_summary(fitted_grid_search)

    best_params = fitted_grid_search.best_params_

    for _, row in df_summary.iterrows():
        df_name = row['Parameter']
        df_value = row['Value']
        
        # make sure the value in the output Dataframe is the same as the expected value as well as the data type.
        expected_value = best_params[df_name]
        assert df_value == expected_value, f"Parameter {param_name} value should be {expected_value}, got {param_value}"
        assert type(df_value) == type(expected_value), f"Parameter {df_name} type should be {type(expected_value)}, got {type(param_value)}"


def test_param_tuning_summary_consistency_across_calls(fitted_grid_search):
    """
    Test that the function will produce identical outputs even after being called for multiple times
    """
    # call the function twice
    df_summary1, best_estimator1 = param_tuning_summary(fitted_grid_search)
    df_summary2, best_estimator2 = param_tuning_summary(fitted_grid_search)
    
    # the output DataFrames should be identical
    pd.testing.assert_frame_equal(df_summary1, df_summary2, check_dtype=True, check_exact=True)
    
    # Estimators should be the same object (not just equal, but identical)
    assert best_estimator1 is best_estimator2, "The estimators returned should be the same."