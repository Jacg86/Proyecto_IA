"""
Decision Tree Model implementation.
Tunes maximum depth, split criteria (gini/entropy), min_samples_split,
and extracts feature importance rankings.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from models.base_model import BaseLEGOModel
import config

class LEGODecisionTree(BaseLEGOModel):
    def __init__(self, criterion='gini', max_depth=14, min_samples_split=4, min_samples_leaf=2, random_state=42):
        super().__init__(name="Decision Tree")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
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

    def get_feature_importances(self):
        return self.model.feature_importances_

def tune_decision_tree(X_train, y_train, X_val, y_val):
    """
    Tunes max_depth and criterion for Decision Tree.
    """
    print("\n--- Evaluando configuraciones de Árbol de Decisión ---")
    candidates = [
        {"criterion": "gini", "max_depth": 8, "min_samples_split": 5},
        {"criterion": "gini", "max_depth": 14, "min_samples_split": 4},
        {"criterion": "entropy", "max_depth": 14, "min_samples_split": 4},
        {"criterion": "gini", "max_depth": 20, "min_samples_split": 2},
        {"criterion": "entropy", "max_depth": 20, "min_samples_split": 2},
        {"criterion": "gini", "max_depth": None, "min_samples_split": 2}
    ]
    
    best_acc = -1.0
    best_cfg = None
    best_model = None
    results = []
    
    for cfg in candidates:
        dt = LEGODecisionTree(
            criterion=cfg["criterion"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"]
        )
        dt.fit(X_train, y_train)
        val_acc = float(np.mean(dt.predict(X_val) == y_val))
        cfg_name = f"crit={cfg['criterion']}, depth={cfg['max_depth']}, split={cfg['min_samples_split']}"
        print(f"  Config: {cfg_name:40s} -> Val Acc: {val_acc*100:.2f}% (Train time: {dt.train_time:.3f}s)")
        results.append({"config": cfg, "val_acc": val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = cfg
            best_model = dt
            
    print(f"\nMejor configuración Árbol: {best_cfg} con Val Acc: {best_acc*100:.2f}%")
    
    # Plot top feature importances
    importances = best_model.get_feature_importances()
    top_indices = np.argsort(importances)[::-1][:20]
    
    out_dir = os.path.join(config.RESULTS_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(top_indices)), importances[top_indices], color='forestgreen', edgecolor='black', alpha=0.85)
    ax.set_title("Top 20 Características Más Importantes (Árbol de Decisión)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Índice de Característica", fontsize=11)
    ax.set_ylabel("Importancia Relativa (Gini / Entropía)", fontsize=11)
    ax.set_xticks(range(len(top_indices)))
    ax.set_xticklabels([f"Feat #{i}" for i in top_indices], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, "decision_tree_feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Gráfico de importancia de características guardado en: {plot_path}")
    
    return best_model, best_cfg, results

