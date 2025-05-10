import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from typing import Optional, Dict, Any


class BaseModel:
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
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted()
        model = self.pipeline.named_steps.get('model')
        if not isinstance(model, ClassifierMixin):
            raise AttributeError(f"{self.model_name} does not support probability estimates.")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y, metrics: Dict[str, Any]):

        self._check_is_fitted()
        y_pred = self.predict(X)
        results = {}
        for name, fn in metrics.items():
            results[name] = fn(y, y_pred)
        return results

    def save(self, path: str):
        joblib.dump(self.pipeline, path)
        return path

    @classmethod
    def load(
            cls,
            path: str,
            model_name: str,
            random_state: Optional[int] = None
    ):
        pipeline = joblib.load(path)
        obj = cls(pipeline=pipeline, model_name=model_name, random_state=random_state)
        obj.is_fitted = True
        return obj

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_name}' is not fitted yet.")
