"""
model_auto_interpret: A package for streamlining machine learning model interpretation.
"""

#__version__ = "0.1.2"

from model_auto_interpret.hyperparameter_tuning_summary import param_tuning_summary
from model_auto_interpret.model_compare import model_cv_metric_compare
from model_auto_interpret.model_evaluation import model_evaluation_plotting
from .__version__ import __version__

__all__ = [
    "param_tuning_summary",
    "model_cv_metric_compare", 
    "model_evaluation_plotting",
    "__version__",
]