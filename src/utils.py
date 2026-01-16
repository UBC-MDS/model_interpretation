import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def create_test_artifacts():
    """
    Creates synthetic data and fitted models for testing ML functions.

    Returns:
        X_train (DataFrame): Training features
        X_test (DataFrame): Test features
        y_train (Series): Training labels (encoded as 'N', 'Y')
        y_test (Series): Test labels (encoded as 'N', 'Y')
        models (dict): Dictionary of fitted pipelines
    
    Example:
        To use this function, simply call:
            X_train, X_test, y_train, y_test, models = create_test_artifacts()
        
        Models included:
            - Dummy: Dummy Classifier
            - SVM: SVM RBF
            - KNN: KNN
            - DecisionTree: Decision Tree
            - RandomForest: Random Forest
        
        To select a specific model for testing, use models dictionary:
            single_model = models["RandomForest"]

        
    """
    # Generate Synthetic Data
    # Create 200 samples with 5 numeric features
    X, y = make_classification(
        n_samples=200, 
        n_features=5, 
        n_informative=3,
        n_redundant=0, 
        random_state=123
    )

    # Wrap in Pandas
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    
    # Map 0/1 to 'N'/'Y' so fbeta_score(pos_label="Y") works
    y_series = pd.Series(y).map({0: 'N', 1: 'Y'})
    y_series.name = "churn"

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=123
    )

    # Define Models and Pipelines
    scaler = StandardScaler()

    models = {
        "Dummy": make_pipeline(scaler, DummyClassifier(strategy="most_frequent")),
        "SVM": make_pipeline(scaler, SVC(kernel="rbf", probability=True, random_state=123)),
        "KNN": make_pipeline(scaler, KNeighborsClassifier(n_neighbors=3)),
        "DecisionTree": make_pipeline(scaler, DecisionTreeClassifier(max_depth=5, random_state=123)),
        "RandomForest": make_pipeline(scaler, RandomForestClassifier(n_estimators=10, random_state=123))
    }

    # Fit the models
    # Test functions expect fitted estimators
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, models
