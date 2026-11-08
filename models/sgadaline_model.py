"""
SGAdaline (Stochastic Gradient Descent Adaline) Model implementation.
Custom implementation of Adaptive Linear Neuron using SGD with MSE loss.
Adapted for 20-class multiclass classification via One-vs-Rest (OvR) strategy.
"""
import time
import numpy as np
from models.base_model import BaseLEGOModel

class BinarySGAdaline:
    """
    Binary Adaline trained with Stochastic Gradient Descent (SGD) and MSE loss.
    Target labels: +1 (positive class) and -1 (negative classes).
    """
    def __init__(self, eta0=0.001, n_iter=60, decay=0.001, random_state=42):
        self.eta0 = eta0
        self.n_iter = n_iter
        self.decay = decay
        self.random_state = random_state
        self.w_ = None
        self.b_ = 0.0
        self.losses_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        # Initialize weights with small random Gaussian values
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0
        self.losses_ = []
        
        for epoch in range(1, self.n_iter + 1):
            # Shuffle indices each epoch
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Learning rate schedule
            eta = self.eta0 / (1.0 + self.decay * epoch)
            epoch_loss = 0.0
            
            for xi, target in zip(X_shuffled, y_shuffled):
                # Linear activation
                net_input = np.dot(xi, self.w_) + self.b_
                error = target - net_input
                # Stochastic weight and bias update
                self.w_ += eta * error * xi
                self.b_ += eta * error
                epoch_loss += 0.5 * (error ** 2)
                
            self.losses_.append(epoch_loss / n_samples)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

class MulticlassSGAdaline(BaseLEGOModel):
    """
    Multiclass SGAdaline using One-vs-Rest (OvR) decomposition for 20 LEGO classes.
    """
    def __init__(self, n_classes=20, eta0=0.001, n_iter=70, decay=0.002, random_state=42):
        super().__init__(name="SGAdaline")
        self.n_classes = n_classes
        self.eta0 = eta0
        self.n_iter = n_iter
        self.decay = decay
        self.random_state = random_state
        self.classifiers_ = []
        self.loss_history_ = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.perf_counter()
        self.classifiers_ = []
        all_epoch_losses = np.zeros(self.n_iter)
        
        classes = np.unique(y_train)
        self.n_classes = len(classes)
        
        for c in range(self.n_classes):
            # Binary target: +1 if class c, else -1
            y_binary = np.where(y_train == c, 1.0, -1.0)
            clf = BinarySGAdaline(
                eta0=self.eta0,
                n_iter=self.n_iter,
                decay=self.decay,
                random_state=self.random_state + c
            )
            clf.fit(X_train, y_binary)
            self.classifiers_.append(clf)
            all_epoch_losses += np.array(clf.losses_)
            
        self.loss_history_ = (all_epoch_losses / self.n_classes).tolist()
        self.train_time = time.perf_counter() - start_time
        return self

    def decision_function(self, X):
        # Shape: (n_samples, n_classes)
        outputs = np.column_stack([clf.net_input(X) for clf in self.classifiers_])
        return outputs

    def predict(self, X):
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        # Softmax for probability calibration
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probas

