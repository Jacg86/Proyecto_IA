"""
Support Vector Machine (SVM) Model implementation.
Evaluates Linear, Polynomial, and RBF kernels, tunes C and gamma,
and provides probability predictions.
"""
import time
import numpy as np
from sklearn.svm import SVC
from models.base_model import BaseLEGOModel

class LEGOSVM(BaseLEGOModel):
    def __init__(self, kernel='rbf', C=5.0, gamma='scale', degree=3, probability=True, random_state=42):
        super().__init__(name=f"SVM ({kernel})")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.random_state = random_state
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.train_time = time.perf_counter() - start_time
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def tune_svm_kernels(X_train, y_train, X_val, y_val):
    """
    Evaluates linear, polynomial, and RBF kernels to find optimal hyperparameters.
    """
    print("\n--- Evaluando configuraciones de Kernel para SVM ---")
    candidates = [
        {"kernel": "linear", "C": 1.0, "gamma": "scale"},
        {"kernel": "linear", "C": 5.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 5.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": 0.01},
        {"kernel": "poly", "C": 5.0, "degree": 3, "gamma": "scale"}
    ]
    
    best_acc = -1.0
    best_config = None
    best_model = None
    results = []
    
    for cfg in candidates:
        svm = LEGOSVM(
            kernel=cfg["kernel"],
            C=cfg["C"],
            gamma=cfg.get("gamma", "scale"),
            degree=cfg.get("degree", 3),
            probability=True
        )
        svm.fit(X_train, y_train)
        preds = svm.predict(X_val)
        acc = float(np.mean(preds == y_val))
        cfg_name = f"kernel={cfg['kernel']}, C={cfg['C']}" + (f", deg={cfg['degree']}" if cfg['kernel']=='poly' else "")
        print(f"  Config: {cfg_name:30s} -> Val Accuracy: {acc*100:.2f}% (Train time: {svm.train_time:.2f}s)")
        results.append({"config": cfg, "val_acc": acc, "train_time": svm.train_time})
        
        if acc > best_acc:
            best_acc = acc
            best_config = cfg
            best_model = svm
            
    print(f"\nMejor configuración SVM: {best_config} con Val Acc: {best_acc*100:.2f}%")
    return best_model, best_config, results

