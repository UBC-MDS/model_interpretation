import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

def model_cv_metric_compare(models_dict, X, y, cv=5):
    """
    Evaluates multiple models using Cross-Validation and returns a metric/scorer comparison DataFrame.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of {model_name: pipeline_object}. 
        Note: Models do not need to be fitted beforehand.
    X : DataFrame
        Features (Training set or full dataset).
    y : Series
        Labels (Training set or full dataset).
    cv : int
        Number of cross-validation folds (default 5).

    Scorers Evaluated
    -----------------
    - accuracy
    - precision (pos_label="Y")
    - recall (pos_label="Y")
    - f1 (pos_label="Y")
    - roc_auc (if model supports predict_proba)

    Returns
    -------
    dataframe : pandas.DataFrame
        Dataframe containing model name and mean evaluation metrics.
    """
    
    # Define Scorers that handle specific pos_label="Y"
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, pos_label="Y"),
        'recall': make_scorer(recall_score, pos_label="Y"),
        'f1': make_scorer(f1_score, pos_label="Y"),
    }

    results_list = []

    for name, model in models_dict.items():
        # Check if model supports probabilities for ROC-AUC
        # We need a separate handling for ROC AUC because it requires probabilities, not just predictions
        current_scorers = scorers.copy()
        
        # Only add ROC_AUC if the model supports predict_proba
        if hasattr(model, "predict_proba"):
            # Note: For string labels, we need to ensure the scorer knows which class is positive.
            # response_method='predict_proba' is handled by make_scorer automatically if configured,
            # but standard 'roc_auc' string in sklearn often assumes 0/1 or specific ordering.
            # We create a custom scorer for safety with string labels.
            def custom_roc(y_true, y_prob):
                 # This helper is needed if y is "Y"/"N" to map it for calculation
                 y_true_num = (y_true == "Y").astype(int)
                 return roc_auc_score(y_true_num, y_prob)
            
            # We tell sklearn to pass the probability of the positive class
            current_scorers['roc_auc'] = make_scorer(custom_roc, response_method="predict_proba")

        # Run Cross-Validation
        cv_results = cross_validate(
            model, 
            X, 
            y, 
            cv=cv, 
            scoring=current_scorers,
            n_jobs=-1 # Use all CPU cores for speed
        )

        # Aggregate Results (Take the Mean of the folds)
        metrics = {"Model": name}
        for metric_name in current_scorers.keys():
            # cross_validate returns keys like 'test_accuracy', 'test_f1', etc.
            key = f"test_{metric_name}"
            if key in cv_results:
                metrics[metric_name] = np.mean(cv_results[key])
            else:
                metrics[metric_name] = np.nan

        results_list.append(metrics)

    # 5. Format Output
    comparison_df = pd.DataFrame(results_list).set_index("Model")
    return comparison_df