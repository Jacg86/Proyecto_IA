"""
Visualization module for evaluation.
Plots confusion matrices, comparative model bar charts,
and visual grids of sample predictions (correct and misclassified).
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def plot_confusion_matrix(cm, model_name, save_filename=None):
    """
    Renders a 20x20 confusion matrix with class names.
    """
    fig, ax = plt.subplots(figsize=(13, 11))
    cm_arr = np.array(cm)
    
    # Normalize by row (true class support) for color scale
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_arr.astype('float') / row_sums
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=config.ALL_CLASSES,
        yticklabels=config.ALL_CLASSES,
        title=f"Matriz de Confusión Normalizada: {model_name}",
        ylabel="Clase Verdadera (Ground Truth)",
        xlabel="Clase Predicha"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    # Text annotation in each cell
    thresh = cm_norm.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            if val > 0:
                ax.text(j, i, f"{val}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=7)
                
    plt.tight_layout()
    if save_filename:
        out_dir = os.path.join(config.RESULTS_DIR, "models")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, save_filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Matriz de confusión guardada en: {path}")
    else:
        plt.show()

def plot_models_comparison(comparison_df, save_filename="models_benchmark_comparison.png"):
    """
    Plots comparison bar charts for the models across:
    1. Accuracy & F1-Score
    2. Training Time (seconds)
    3. Inference Latency (ms / sample)
    """
    out_dir = os.path.join(config.RESULTS_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = comparison_df["Model"].tolist()
    accuracies = [acc * 100 for acc in comparison_df["Accuracy"].tolist()]
    f1_scores = [f1 * 100 for f1 in comparison_df["F1-Score (Macro)"].tolist()]
    train_times = comparison_df["Train Time (s)"].tolist()
    infer_latencies = comparison_df["Inference Latency (ms)"].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    # 1. Accuracy & F1-Score
    axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4', edgecolor='black')
    axes[0].bar(x + width/2, f1_scores, width, label='F1-Score (Macro)', color='#2ca02c', edgecolor='black')
    axes[0].set_title("Rendimiento Clasificación (%)", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    axes[0].set_ylabel("Porcentaje (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Training Time
    bars_t = axes[1].bar(x, train_times, color='#ff7f0e', edgecolor='black', alpha=0.85)
    axes[1].set_title("Tiempo de Entrenamiento (segundos)", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    axes[1].set_ylabel("Segundos", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for b in bars_t:
        yv = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2.0, yv + 0.1, f"{yv:.1f}s", ha='center', va='bottom', fontsize=8)
        
    # 3. Inference Latency
    bars_i = axes[2].bar(x, infer_latencies, color='#9467bd', edgecolor='black', alpha=0.85)
    axes[2].set_title("Latencia de Inferencia (ms/muestra)", fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    axes[2].set_ylabel("Milisegundos", fontsize=11)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    for b in bars_i:
        yv = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2.0, yv + 0.02, f"{yv:.2f}ms", ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()
    path = os.path.join(out_dir, save_filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Gráfico comparativo de modelos guardado en: {path}")

def plot_scenario_comparison(shape_scores, color_scores, save_filename="shape_vs_color_comparison.png"):
    """
    Plots the comparative performance of Shape Only vs Shape + Color.
    Directly answers: 'Does color help or introduce bias?'
    """
    out_dir = os.path.join(config.RESULTS_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    
    models = list(shape_scores.keys())
    s_acc = [shape_scores[m] * 100 for m in models]
    c_acc = [color_scores[m] * 100 for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, s_acc, width, label='Solo Forma y Geometría', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, c_acc, width, label='Forma + Geometría + Color (HSV)', color='#e67e22', edgecolor='black')
    
    ax.set_title("Comparación Experimental: Solo Forma vs. Forma + Color", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Exactitud (Accuracy %)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate differences
    for i in range(len(models)):
        diff = c_acc[i] - s_acc[i]
        sign = "+" if diff >= 0 else ""
        ax.text(x[i], max(s_acc[i], c_acc[i]) + 2, f"{sign}{diff:.1f}%", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    path = os.path.join(out_dir, save_filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Gráfico de Forma vs Color guardado en: {path}")

