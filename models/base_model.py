"""
Base model class definition.
Standardizes fit, predict, predict_proba, time tracking, and persistence interfaces.
"""
import time
import joblib
import numpy as np

class BaseLEGOModel:
    def __init__(self, name="BaseModel"):
        self.name = name
        self.train_time = 0.0
        self.inference_time_per_sample = 0.0

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def measure_inference_speed(self, X, num_runs=3):
        """
        Measures inference time in milliseconds per sample.
        """
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = self.predict(X)
        total_time = (time.perf_counter() - start) / num_runs
        self.inference_time_per_sample = (total_time / len(X)) * 1000.0  # ms/sample
        return self.inference_time_per_sample

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)

