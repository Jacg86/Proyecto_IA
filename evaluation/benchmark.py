"""
Benchmark execution module.
Trains and evaluates all 6 models across identical train/val/test splits,
computes metrics, generates confusion matrices, and writes comparative reports.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import create_or_load_splits
from features.feature_extractor import prepare_and_cache_features
from evaluation.metrics import compute_classification_metrics
from evaluation.visualizer import plot_confusion_matrix, plot_models_comparison, plot_scenario_comparison

from models.perceptron_model import LEGOPerceptron
from models.sgadaline_model import MulticlassSGAdaline
from models.svm_model import LEGOSVM, tune_svm_kernels
from models.decision_tree_model import LEGODecisionTree, tune_decision_tree
from models.neural_net_model import LEGONeuralNetwork
from models.cnn_model import LEGOCNN

def run_full_benchmark():
    print("=" * 70)
    print("ETAPAS 5 Y 6: ENTRENAMIENTO Y EVALUACIÓN EXPERIMENTAL DE LOS 6 MODELOS")
    print("=" * 70)
    
    checkpoints_dir = os.path.join(config.MODELS_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Load splits and precomputed features
    splits = create_or_load_splits()
    feat_shape, feat_all = prepare_and_cache_features()
    
    X_tr_shape, y_tr = feat_shape["X_train"], feat_shape["y_train"]
    X_va_shape, y_va = feat_shape["X_val"], feat_shape["y_val"]
    X_te_shape, y_te = feat_shape["X_test"], feat_shape["y_test"]
    
    X_tr_all = feat_all["X_train"]
    X_va_all = feat_all["X_val"]
    X_te_all = feat_all["X_test"]
    
    print(f"\nDatos listos para experimentación:")
    print(f"  Train: {len(X_tr_all)} muestras | Val: {len(X_va_all)} muestras | Test: {len(X_te_all)} muestras")
    
    # Track results
    all_results = []
    shape_scores = {}
    color_scores = {}
    trained_models = {}
    
    # =========================================================================
    # MODEL 1: PERCEPTRÓN
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. MODELO: PERCEPTRÓN")
    print("=" * 60)
    
    # Scenario A: Shape only
    p_shape = LEGOPerceptron(max_iter=2000, eta0=0.01)
    p_shape.fit(X_tr_shape, y_tr)
    acc_p_shape = float(np.mean(p_shape.predict(X_te_shape) == y_te))
    shape_scores["Perceptrón"] = acc_p_shape
    
    # Scenario B: Shape + Color
    p_all = LEGOPerceptron(max_iter=2000, eta0=0.01)
    p_all.fit(X_tr_all, y_tr)
    infer_ms_p = p_all.measure_inference_speed(X_te_all)
    preds_p = p_all.predict(X_te_all)
    acc_p_all = float(np.mean(preds_p == y_te))
    color_scores["Perceptrón"] = acc_p_all
    
    m_p = compute_classification_metrics(y_te, preds_p, "Perceptrón", p_all.train_time, infer_ms_p)
    all_results.append(m_p)
    trained_models["perceptron"] = p_all
    p_all.save(os.path.join(checkpoints_dir, "perceptron_model.joblib"))
    plot_confusion_matrix(m_p["confusion_matrix"], "Perceptrón", "cm_perceptron.png")
    print(f"Perceptrón Test Acc (Forma): {acc_p_shape*100:.2f}% | (Forma+Color): {acc_p_all*100:.2f}% | F1: {m_p['f1_macro']*100:.2f}%")

    # =========================================================================
    # MODEL 2: SGADALINE
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. MODELO: SGADALINE (Adaline entrenado con SGD)")
    print("=" * 60)
    
    # Scenario A: Shape only
    adaline_shape = MulticlassSGAdaline(eta0=0.0005, n_iter=60, decay=0.001)
    adaline_shape.fit(X_tr_shape, y_tr)
    acc_ada_shape = float(np.mean(adaline_shape.predict(X_te_shape) == y_te))
    shape_scores["SGAdaline"] = acc_ada_shape
    
    # Scenario B: Shape + Color
    adaline_all = MulticlassSGAdaline(eta0=0.0005, n_iter=60, decay=0.001)
    adaline_all.fit(X_tr_all, y_tr)
    infer_ms_ada = adaline_all.measure_inference_speed(X_te_all)
    preds_ada = adaline_all.predict(X_te_all)
    acc_ada_all = float(np.mean(preds_ada == y_te))
    color_scores["SGAdaline"] = acc_ada_all
    
    m_ada = compute_classification_metrics(y_te, preds_ada, "SGAdaline", adaline_all.train_time, infer_ms_ada)
    all_results.append(m_ada)
    trained_models["sgadaline"] = adaline_all
    adaline_all.save(os.path.join(checkpoints_dir, "sgadaline_model.joblib"))
    plot_confusion_matrix(m_ada["confusion_matrix"], "SGAdaline", "cm_sgadaline.png")
    print(f"SGAdaline Test Acc (Forma): {acc_ada_shape*100:.2f}% | (Forma+Color): {acc_ada_all*100:.2f}% | F1: {m_ada['f1_macro']*100:.2f}%")

    # =========================================================================
    # MODEL 3: SUPPORT VECTOR MACHINE (SVM)
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. MODELO: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60)
    
    # Kernel tuning on validation set
    best_svm, best_svm_cfg, _ = tune_svm_kernels(X_tr_all, y_tr, X_va_all, y_va)
    infer_ms_svm = best_svm.measure_inference_speed(X_te_all)
    preds_svm = best_svm.predict(X_te_all)
    acc_svm_all = float(np.mean(preds_svm == y_te))
    color_scores["SVM"] = acc_svm_all
    
    # Test on shape only with best kernel
    svm_shape = LEGOSVM(kernel=best_svm_cfg["kernel"], C=best_svm_cfg["C"], degree=best_svm_cfg.get("degree", 3))
    svm_shape.fit(X_tr_shape, y_tr)
    acc_svm_shape = float(np.mean(svm_shape.predict(X_te_shape) == y_te))
    shape_scores["SVM"] = acc_svm_shape
    
    m_svm = compute_classification_metrics(y_te, preds_svm, f"SVM ({best_svm_cfg['kernel']})", best_svm.train_time, infer_ms_svm)
    all_results.append(m_svm)
    trained_models["svm"] = best_svm
    best_svm.save(os.path.join(checkpoints_dir, "svm_model.joblib"))
    plot_confusion_matrix(m_svm["confusion_matrix"], f"SVM ({best_svm_cfg['kernel']})", "cm_svm.png")
    print(f"SVM Test Acc (Forma): {acc_svm_shape*100:.2f}% | (Forma+Color): {acc_svm_all*100:.2f}% | F1: {m_svm['f1_macro']*100:.2f}%")

    # =========================================================================
    # MODEL 4: ÁRBOLES DE DECISIÓN
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. MODELO: ÁRBOLES DE DECISIÓN")
    print("=" * 60)
    
    best_dt, best_dt_cfg, _ = tune_decision_tree(X_tr_all, y_tr, X_va_all, y_va)
    infer_ms_dt = best_dt.measure_inference_speed(X_te_all)
    preds_dt = best_dt.predict(X_te_all)
    acc_dt_all = float(np.mean(preds_dt == y_te))
    color_scores["Árbol de Decisión"] = acc_dt_all
    
    dt_shape = LEGODecisionTree(criterion=best_dt_cfg["criterion"], max_depth=best_dt_cfg["max_depth"], min_samples_split=best_dt_cfg["min_samples_split"])
    dt_shape.fit(X_tr_shape, y_tr)
    acc_dt_shape = float(np.mean(dt_shape.predict(X_te_shape) == y_te))
    shape_scores["Árbol de Decisión"] = acc_dt_shape
    
    m_dt = compute_classification_metrics(y_te, preds_dt, "Árbol de Decisión", best_dt.train_time, infer_ms_dt)
    all_results.append(m_dt)
    trained_models["decision_tree"] = best_dt
    best_dt.save(os.path.join(checkpoints_dir, "decision_tree_model.joblib"))
    plot_confusion_matrix(m_dt["confusion_matrix"], "Árbol de Decisión", "cm_decision_tree.png")
    print(f"Árbol Test Acc (Forma): {acc_dt_shape*100:.2f}% | (Forma+Color): {acc_dt_all*100:.2f}% | F1: {m_dt['f1_macro']*100:.2f}%")

    # =========================================================================
    # MODEL 5: REDES NEURONALES ARTIFICIALES (ANN / MLP)
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. MODELO: RED NEURONAL ARTIFICIAL (ANN / MLP)")
    print("=" * 60)
    
    mlp_all = LEGONeuralNetwork(input_dim=X_tr_all.shape[1], epochs=35, batch_size=64, lr=0.001)
    mlp_all.fit(X_tr_all, y_tr, X_va_all, y_va)
    infer_ms_mlp = mlp_all.measure_inference_speed(X_te_all)
    preds_mlp = mlp_all.predict(X_te_all)
    acc_mlp_all = float(np.mean(preds_mlp == y_te))
    color_scores["Red Neuronal (ANN)"] = acc_mlp_all
    
    mlp_shape = LEGONeuralNetwork(input_dim=X_tr_shape.shape[1], epochs=30, batch_size=64, lr=0.001)
    mlp_shape.fit(X_tr_shape, y_tr, X_va_shape, y_va)
    acc_mlp_shape = float(np.mean(mlp_shape.predict(X_te_shape) == y_te))
    shape_scores["Red Neuronal (ANN)"] = acc_mlp_shape
    
    m_mlp = compute_classification_metrics(y_te, preds_mlp, "Red Neuronal (ANN)", mlp_all.train_time, infer_ms_mlp)
    all_results.append(m_mlp)
    trained_models["neural_network"] = mlp_all
    torch.save(mlp_all.net.state_dict(), os.path.join(checkpoints_dir, "ann_model.pth"))
    plot_confusion_matrix(m_mlp["confusion_matrix"], "Red Neuronal (ANN)", "cm_ann.png")
    print(f"ANN Test Acc (Forma): {acc_mlp_shape*100:.2f}% | (Forma+Color): {acc_mlp_all*100:.2f}% | F1: {m_mlp['f1_macro']*100:.2f}%")

    # =========================================================================
    # MODEL 6: REDES NEURONALES CONVOLUCIONALES (CNN)
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. MODELO: RED NEURONAL CONVOLUCIONAL (CNN)")
    print("=" * 60)
    
    cnn = LEGOCNN(epochs=25, batch_size=64, lr=0.001)
    cnn.fit_from_splits(splits["train"], splits["val"])
    
    # Measure CNN inference latency
    start_infer = time.perf_counter()
    preds_cnn = cnn.predict_records(splits["test"])
    infer_ms_cnn = ((time.perf_counter() - start_infer) / len(splits["test"])) * 1000.0
    acc_cnn = float(np.mean(preds_cnn == y_te))
    
    m_cnn = compute_classification_metrics(y_te, preds_cnn, "CNN", cnn.train_time, infer_ms_cnn)
    all_results.append(m_cnn)
    trained_models["cnn"] = cnn
    torch.save(cnn.net.state_dict(), os.path.join(checkpoints_dir, "cnn_model.pth"))
    plot_confusion_matrix(m_cnn["confusion_matrix"], "CNN", "cm_cnn.png")
    print(f"CNN Test Accuracy: {acc_cnn*100:.2f}% | F1: {m_cnn['f1_macro']*100:.2f}% | Train Time: {cnn.train_time:.1f}s | Latency: {infer_ms_cnn:.2f}ms/sample")

    # =========================================================================
    # SUMMARY & COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 7: COMPARACIÓN Y SELECCIÓN DEL MEJOR MODELO")
    print("=" * 70)
    
    # Create comparison table DataFrame
    rows = []
    for m in all_results:
        rows.append({
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Precision (Macro)": m["precision_macro"],
            "Recall (Macro)": m["recall_macro"],
            "F1-Score (Macro)": m["f1_macro"],
            "F1-Score (Weighted)": m["f1_weighted"],
            "Train Time (s)": m["train_time_sec"],
            "Inference Latency (ms)": m["inference_ms_sample"]
        })
        
    df_comp = pd.DataFrame(rows)
    print(df_comp.to_string(index=False))
    
    # Save CSV and JSON
    out_dir = os.path.join(config.RESULTS_DIR, "models")
    df_comp.to_csv(os.path.join(out_dir, "models_comparison_table.csv"), index=False)
    
    with open(os.path.join(out_dir, "all_benchmark_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
        
    # Generate Visual Comparison Plots
    plot_models_comparison(df_comp, "models_benchmark_comparison.png")
    plot_scenario_comparison(shape_scores, color_scores, "shape_vs_color_comparison.png")
    
    # Determine winner
    winner = df_comp.sort_values(by="F1-Score (Macro)", ascending=False).iloc[0]
    print(f"\n=======================================================")
    print(f"MODELO GANADOR: {winner['Model']}")
    print(f"Accuracy: {winner['Accuracy']*100:.2f}% | F1-Macro: {winner['F1-Score (Macro)']*100:.2f}%")
    print(f"=======================================================")
    
    return df_comp, trained_models

if __name__ == "__main__":
    run_full_benchmark()

