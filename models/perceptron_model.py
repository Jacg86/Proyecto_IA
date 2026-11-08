"""
Perceptron Model implementation.
Linear threshold classifier using One-vs-Rest for 20-class classification.
Calibrated for probabilistic output.
"""
import time
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV

from models.base_model import BaseLEGOModel

class LEGOPerceptron(BaseLEGOModel):
    def __init__(self, max_iter=2000, tol=1e-3, eta0=0.01, random_state=42):
        super().__init__(name="Perceptron")
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.random_state = random_state
        self.base_perceptron = Perceptron(
            max_iter=self.max_iter,
            tol=self.tol,
            eta0=self.eta0,
            random_state=self.random_state,
            penalty='l2',
            alpha=1e-4
        )
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.perf_counter()
        # Wrap in CalibratedClassifierCV to obtain calibrated class probabilities
        calibrated = CalibratedClassifierCV(estimator=self.base_perceptron, method='sigmoid', cv=3)
        calibrated.fit(X_train, y_train)
        self.model = calibrated
        self.train_time = time.perf_counter() - start_time
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

