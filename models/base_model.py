"""
This module defines the BaseModel class which serves as a foundation for all machine learning models
in the project. The class provides a consistent interface for fitting, predicting, evaluating,
and saving/loading models.

Key features:
- Consistent pipeline-based architecture
- Common interface for model evaluation
- Model persistence through save/load functionality
- Type checking and runtime validation
"""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from typing import Optional, Dict, Any


class BaseModel:
    """
    This class wraps a scikit-learn Pipeline and provides a consistent interface
    for model operations. All specific models should inherit from this class
    for consistency across the project.

    Attributes:
        pipeline: The scikit-learn Pipeline containing preprocessing and model
        model_name: Name identifier for the model
        random_state: Seed for reproducibility
        is_fitted: Flag indicating if the model has been fitted
    """

    def __init__(
            self,
            pipeline: Pipeline,
            model_name: str,
            random_state: Optional[int] = None
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.random_state = random_state
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the model pipeline to training data.

        Args:
            X: Feature data
            y: Target values

        Returns:
            self: The fitted model instance
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Generate predictions for given data.

        Args:
            X: Feature data

        Returns:
            Predicted classes
        """
        self._check_is_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Generate class probabilities for given data.

        Raises:
            AttributeError: If the model doesn't support probability estimates

        Returns:
            Class probabilities
        """
        self._check_is_fitted()
        model = self.pipeline.named_steps.get('model')
        if not isinstance(model, ClassifierMixin):
            raise AttributeError(f"{self.model_name} does not support probability estimates.")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y, metrics: Dict[str, Any]):
        """
        Evaluate model performance using provided metrics.

        Args:
            X: Feature data
            y: True labels
            metrics: Dictionary of metric functions

        Returns:
            Dictionary of metric results
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        results = {}
        for name, fn in metrics.items():
            results[name] = fn(y, y_pred)
        return results

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: File path to save model

        Returns:
            Save path
        """
        joblib.dump(self.pipeline, path)
        return path

    @classmethod
    def load(
            cls,
            path: str,
            model_name: str,
            random_state: Optional[int] = None
    ):
        """
        Load model from disk.

        This class method creates a new model instance from a saved pipeline.

        Args:
            path: Path to saved model file
            model_name: Name for the loaded model
            random_state: Seed for reproducibility

        Returns:
            BaseModel: Loaded model instance
        """
        pipeline = joblib.load(path)
        obj = cls(pipeline=pipeline, model_name=model_name, random_state=random_state)
        obj.is_fitted = True
        return obj

    def _check_is_fitted(self):
        """
        Verify model is fitted before prediction.

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_name}' is not fitted yet.")