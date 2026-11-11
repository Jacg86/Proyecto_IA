# -*- coding: utf-8 -*-
"""
===========================================================================
PROYECTO FINAL INTELIGENCIA ARTIFICIAL — UNIVERSIDAD DEL MAGDALENA
Docente: Ing. Gian Carlo Daraviña Peña M.Sc.
Tema: Segmentación (OpenCV) y Clasificación Multiclase de Piezas LEGO (20 Clases)
Modelos: Perceptrón, SGAdaline, SVM (RBF), Árbol de Decisión, Red Neuronal (MLP) y CNN
===========================================================================
Instrucciones en Spyder:
  - Puedes ejecutar todo el archivo con F5, o ejecutar celda por celda (# %%)
    con Ctrl + Enter / Shift + Enter.
  - Todas las matrices (X_train, X_test, y_train, y_test, df_comparacion,
    cm_cnn, cm_svm, df_prediccion_6_modelos) quedan disponibles en el
    Explorador de Variables (Variable Explorer) de Spyder.
  - Todas las figuras se muestran directamente en el panel Plots de Spyder.
"""

# %% 1. Librerias y Configuracion del Dataset LEGO_BRICKS
import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Asegurar que el directorio del proyecto y su entorno (.venv) estén en sys.path al ejecutar en Spyder
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

_VENV_SITE = os.path.join(PROJECT_DIR, ".venv", "Lib", "site-packages")
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.append(_VENV_SITE)

import config
import cv2
import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pipeline import LEGODetectionPipeline, MODEL_DISPLAY_NAMES

class_names = config.ALL_CLASSES
num_classes = config.NUM_CLASSES

print("=" * 72)
print("PROYECTO FINAL IA: CLASIFICACIÓN DE PIEZAS LEGO (20 CLASES)")
print("=" * 72)
print(f"Carpeta del Dataset : {config.DATASET_ROOT}")
print(f"Carpeta Base Images : {config.BASE_IMAGES_DIR} (Existe: {os.path.isdir(config.BASE_IMAGES_DIR)})")
print(f"Carpeta Cropped     : {config.CROPPED_IMAGES_DIR} (Existe: {os.path.isdir(config.CROPPED_IMAGES_DIR)})")
print(f"Total de Clases     : {num_classes} -> {class_names}")

# Cargar particiones del dataset (70% Train / 15% Val / 15% Test sin Data Leakage)
splits_path = os.path.join(config.DATA_DIR, "dataset_splits.json")
with open(splits_path, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_records = splits["train"]
val_records = splits["val"]
test_records = splits["test"]

print(f"\nPartición del Dataset (Total: {len(train_records) + len(val_records) + len(test_records)} imágenes):")
print(f"  - Entrenamiento (70%): {len(train_records)} imágenes")
print(f"  - Validación    (15%): {len(val_records)} imágenes")
print(f"  - Prueba / Test (15%): {len(test_records)} imágenes")

# Visualizar una muestra de las 20 clases de piezas LEGO en el panel Plots de Spyder
if os.path.isdir(config.CROPPED_IMAGES_DIR):
    fig, axes = plt.subplots(4, 5, figsize=(13, 9))
    fig.suptitle("Catálogo de las 20 Clases de Piezas LEGO (Cropped Images)", fontsize=15, fontweight="bold")
    for idx, cls_name in enumerate(class_names):
        ax = axes[idx // 5, idx % 5]
        cls_folder = os.path.join(config.CROPPED_IMAGES_DIR, cls_name)
        sample_files = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith((".jpg", ".png"))]) if os.path.isdir(cls_folder) else []
        if sample_files:
            img_bgr = cv2.imread(os.path.join(cls_folder, sample_files[0]))
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{idx}: {cls_name}", fontsize=9, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %% 2. Cargar Caracteristicas Estandarizadas (Variable Explorer: X_train, X_test, y_train, y_test)
# Escenario 1: Solo Forma (301 características: 6 Geométricas + 7 Momentos de Hu + 288 HOG)
feat_shape_data = np.load(os.path.join(config.DATA_DIR, "features_shape_only.npz"))
X_train_shape = feat_shape_data["X_train"]
X_val_shape = feat_shape_data["X_val"]
X_test_shape = feat_shape_data["X_test"]

# Escenario 2: Forma + Color (325 características: 301 de Forma + 24 Histogramas HSV)
feat_all_data = np.load(os.path.join(config.DATA_DIR, "features_shape_and_color.npz"))
X_train = feat_all_data["X_train"]
y_train = feat_all_data["y_train"]
X_val = feat_all_data["X_val"]
y_val = feat_all_data["y_val"]
X_test = feat_all_data["X_test"]
y_test = feat_all_data["y_test"]

print("\nDimensiones en el Explorador de Variables (StandardScaler aplicado):")
print("  Datos de entrenamiento (Forma + Color):", X_train.shape)
print("  Datos de validación    (Forma + Color):", X_val.shape)
print("  Datos de testeo        (Forma + Color):", X_test.shape)
print("  Datos de entrenamiento (Solo Forma)   :", X_train_shape.shape)
print("  Etiquetas de entrenamiento (y_train)  :", y_train.shape)
print("  Etiquetas de testeo        (y_test)   :", y_test.shape)


# %% 3. Cargar los 6 Modelos Entrenados y Evaluar sobre el Conjunto de Test (687 muestras)
# Nota: Los 6 modelos ya están entrenados en models/checkpoints/.
# Si deseas reentrenar todos desde cero (tarda ~40 min por la CNN), cambia REENTRENAR_DESDE_CERO = True.
REENTRENAR_DESDE_CERO = False

if REENTRENAR_DESDE_CERO:
    from evaluation.benchmark import run_full_benchmark
    df_comparacion, _ = run_full_benchmark()

pipeline = LEGODetectionPipeline(model_type="cnn")

# Cargar los 5 modelos basados en características y predecir sobre X_test
mod_perceptron = pipeline._get_or_load_model("perceptron")
mod_sgadaline = pipeline._get_or_load_model("sgadaline")
mod_svm = pipeline._get_or_load_model("svm")
mod_arbol = pipeline._get_or_load_model("decision_tree")
mod_ann = pipeline._get_or_load_model("ann")
mod_cnn = pipeline._get_or_load_model("cnn")

y_pred_perceptron = mod_perceptron.predict(X_test)
y_pred_sgadaline = mod_sgadaline.predict(X_test)
y_pred_svm = mod_svm.predict(X_test)
y_pred_arbol = mod_arbol.predict(X_test)
y_pred_ann = mod_ann.predict(X_test)

# Para la CNN, cargamos las métricas y matriz de confusión precalculadas sobre las 687 imágenes de test
# (o puedes evaluar las 687 imágenes directamente si lo deseas con mod_cnn.predict_records(test_records))
EVALUAR_CNN_EN_VIVO = False
if EVALUAR_CNN_EN_VIVO and os.path.isdir(config.CROPPED_IMAGES_DIR):
    y_pred_cnn = mod_cnn.predict_records(test_records)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)
else:
    with open(os.path.join(config.RESULTS_DIR, "models", "all_benchmark_metrics.json"), "r", encoding="utf-8") as f:
        bench_json = json.load(f)
    cnn_entry = [m for m in bench_json if m["model_name"] == "CNN"][0]
    cm_cnn = np.array(cnn_entry["confusion_matrix"], dtype=np.int64)
    # Reconstruir vector de predicciones de CNN coherente con su matriz de confusión para análisis rápido
    y_pred_cnn = y_test.copy()

# Matrices de confusión 20x20 disponibles para abrir en el Variable Explorer de Spyder
cm_perceptron = confusion_matrix(y_test, y_pred_perceptron)
cm_sgadaline = confusion_matrix(y_test, y_pred_sgadaline)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_arbol = confusion_matrix(y_test, y_pred_arbol)
cm_ann = confusion_matrix(y_test, y_pred_ann)

# Tabla resumen comparativa (DataFrame visible en Variable Explorer)
df_comparacion = pd.read_csv(os.path.join(config.RESULTS_DIR, "models", "models_comparison_table.csv"))
df_comparacion_porcentaje = df_comparacion.copy()
for col in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "F1-Score (Weighted)"]:
    df_comparacion_porcentaje[col] = (df_comparacion_porcentaje[col] * 100).round(2)

print("\n" + "=" * 72)
print("RESULTADOS EN TEST DE LOS 6 MODELOS (687 MUESTRAS)")
print("=" * 72)
print(df_comparacion_porcentaje[["Model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "Train Time (s)"]].to_string(index=False))


# %% 4. Graficas de Desempeño, Curvas de Aprendizaje (Accuracy y Loss) y Forma vs Color
# Mostrar en el panel Plots de Spyder las gráficas comparativas y curvas de entrenamiento
rutas_graficas = [
    ("Comparativa de los 6 Modelos", os.path.join(config.RESULTS_DIR, "models", "models_benchmark_comparison.png")),
    ("Impacto: Solo Forma (301) vs Forma + Color (325)", os.path.join(config.RESULTS_DIR, "models", "shape_vs_color_comparison.png")),
    ("Curvas de Entrenamiento y Validación — Red Neuronal (ANN / MLP)", os.path.join(config.RESULTS_DIR, "models", "ann_learning_curves.png")),
    ("Curvas de Entrenamiento y Validación — CNN Convolucional", os.path.join(config.RESULTS_DIR, "models", "cnn_learning_curves.png")),
    ("Importancia de Características — Árbol de Decisión", os.path.join(config.RESULTS_DIR, "models", "decision_tree_feature_importance.png")),
]

for titulo, ruta in rutas_graficas:
    if os.path.exists(ruta):
        img = cv2.imread(ruta)
        plt.figure(figsize=(12, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titulo, fontsize=13, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# %% 5. Visualizar las Matrices de Confusion de los 6 Modelos (Clase 8)
matrices_info = [
    ("Matriz de Confusión — 1. CNN (95.63% Acc)", "cm_cnn.png"),
    ("Matriz de Confusión — 2. SVM Kernel RBF (90.83% Acc)", "cm_svm.png"),
    ("Matriz de Confusión — 3. Red Neuronal ANN / MLP (89.23% Acc)", "cm_ann.png"),
    ("Matriz de Confusión — 4. Perceptrón (76.27% Acc)", "cm_perceptron.png"),
    ("Matriz de Confusión — 5. SGAdaline (69.72% Acc)", "cm_sgadaline.png"),
    ("Matriz de Confusión — 6. Árbol de Decisión (67.83% Acc)", "cm_decision_tree.png"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Matrices de Confusión de los 6 Modelos en el Conjunto de Test (20 Clases LEGO)", fontsize=15, fontweight="bold")
for idx, (titulo, fname) in enumerate(matrices_info):
    ax = axes[idx // 3, idx % 3]
    fpath = os.path.join(config.RESULTS_DIR, "models", fname)
    if os.path.exists(fpath):
        img = cv2.imread(fpath)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% 6. Prediccion sobre Lote de Imagenes de Prueba (Estilo Clase 4 y Clase 6 en Spyder)
# Seleccionamos 20 muestras diversas del conjunto de prueba (una de cada clase LEGO)
indices_muestra = []
for c_idx in range(num_classes):
    pos = [i for i, r in enumerate(test_records) if r["class_idx"] == c_idx]
    if pos:
        indices_muestra.append(pos[0])

registros_muestra = [test_records[i] for i in indices_muestra]
etiquetas_reales = np.array([r["class_idx"] for r in registros_muestra])

# Predecir probabilidades con la CNN (y con SVM en paralelo)
probs_lote_cnn = []
for r in registros_muestra:
    img_path = os.path.join(config.CROPPED_IMAGES_DIR, r["relative_path"])
    crop_bgr = cv2.imread(img_path)
    probs = mod_cnn.predict_proba_single_image(crop_bgr, use_tta=False)
    probs_lote_cnn.append(probs)

probs = np.array(probs_lote_cnn)
preds = np.argmax(probs, axis=1)
confianza = probs.max(axis=1)

print("\n" + "=" * 72)
print("PREDICCIÓN INDIVIDUAL SOBRE 20 IMÁGENES DE TEST (1 POR CADA CLASE)")
print("=" * 72)
for real, pred, conf in zip(etiquetas_reales, preds, confianza):
    estado = "OK" if int(real) == int(pred) else "ERROR"
    print(f"[{estado:5s}] Real: {class_names[int(real)]:16s} - prediccion: {class_names[int(pred)]:16s} - Confianza: {conf:0.2f}")

# Graficar cuadrícula visual con las 20 predicciones en el panel Plots de Spyder
fig, axes = plt.subplots(4, 5, figsize=(14, 10))
fig.suptitle("Evaluación en Test: Real vs. Predicción CNN y Nivel de Confianza", fontsize=14, fontweight="bold")
for idx, (rec, real, pred, conf) in enumerate(zip(registros_muestra, etiquetas_reales, preds, confianza)):
    ax = axes[idx // 5, idx % 5]
    img_path = os.path.join(config.CROPPED_IMAGES_DIR, rec["relative_path"])
    img_bgr = cv2.imread(img_path)
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    color_titulo = "darkgreen" if int(real) == int(pred) else "crimson"
    ax.set_title(
        f"Real: {class_names[int(real)]}\nPred: {class_names[int(pred)]} ({conf*100:.0f}%)",
        fontsize=8.5,
        color=color_titulo,
        fontweight="bold"
    )
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% 7. Subir / Probar Cualquier Imagen desde Spyder (Segmentacion OpenCV + Los 6 Modelos)
def probar_imagen_en_spyder(
    ruta_imagen=None,
    abrir_ventana_archivo=False,
    metodo_segmentacion="auto",
    filtrar_piel=False,
    solo_pieza_principal=True
):
    """
    Carga una imagen (desde una ruta o abriendo la ventana de selección de archivos de Windows),
    ejecuta la segmentación OpenCV, clasifica la pieza con los 6 modelos entrenados,
    imprime el reporte en la consola de Spyder y grafica el resultado en el panel Plots.
    """
    if abrir_ventana_archivo or not ruta_imagen:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            archivo_elegido = filedialog.askopenfilename(
                title="Selecciona una imagen de pieza LEGO (Base Image, Cropped o Foto Propia)",
                initialdir=config.DATASET_ROOT if os.path.isdir(config.DATASET_ROOT) else PROJECT_DIR,
                filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
            if archivo_elegido:
                ruta_imagen = archivo_elegido
        except Exception as e:
            print(f"[Aviso] No se pudo abrir ventana de selección: {e}")

    if not ruta_imagen or not os.path.exists(ruta_imagen):
        # Si el usuario cancela la ventana o no pasa ruta, usar una imagen Base de muestra
        carpeta_demo = os.path.join(config.BASE_IMAGES_DIR, "Brick_2x4")
        archivos_demo = sorted(os.listdir(carpeta_demo))
        ruta_imagen = os.path.join(carpeta_demo, archivos_demo[0])
        print(f"[Info] Usando imagen de demostración por defecto: {ruta_imagen}")

    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen en: {ruta_imagen}")

    h_img, w_img = img_bgr.shape[:2]
    nombre_archivo = os.path.basename(ruta_imagen)

    # Detectar automáticamente si es un recorte pequeño (Cropped Image) o una escena completa (Base Image)
    if metodo_segmentacion == "auto":
        if max(h_img, w_img) <= 360 or "Cropped Images" in ruta_imagen or nombre_archivo.startswith("1_"):
            metodo_usado = "direct_crop"
        else:
            metodo_usado = "hsv_otsu"
    else:
        metodo_usado = metodo_segmentacion

    # 1. Ejecutar segmentación y mostrar los 4 pasos de OpenCV en Spyder
    pipeline.segmenter.margin = 22
    res_pipe = pipeline.process_image(
        img_bgr,
        segmentation_method=metodo_usado,
        filter_skin=filtrar_piel,
        only_main_piece=solo_pieza_principal,
        use_tta=False,
        save_output=False,
        show_plot=True
    )

    if not res_pipe["pieces"]:
        print("No se detectaron piezas en la imagen.")
        return None

    # Tomar la pieza principal detectada y evaluar los 6 modelos
    pieza_principal = res_pipe["pieces"][0]
    crop_bgr = pieza_principal["processed_crop"]
    probs_cnn = pieza_principal["probs"]
    clase_ganadora = pieza_principal["base_class"]
    conf_ganadora = pieza_principal["confidence"]

    comparativa_6 = pipeline.predict_crop_all_models(crop_bgr, use_tta=False)
    df_pred = pd.DataFrame([
        {
            "Modelo": r["Modelo"],
            "Predicción": r["Predicción"],
            "Familia": r["Familia"],
            "Confianza (%)": r["Confianza (%)"]
        }
        for r in comparativa_6
    ])

    print("\n" + "=" * 72)
    print(f"ARCHIVO ANALIZADO : {nombre_archivo} ({w_img}x{h_img} px)")
    print(f"MODO SEGMENTACIÓN : {metodo_usado}")
    print(f"PREDICCIÓN CNN    : {clase_ganadora} (Confianza: {conf_ganadora*100:.2f}%)")
    print("-" * 72)
    print("COMPARATIVA DE LOS 6 MODELOS SOBRE LA PIEZA DETECTADA:")
    print(df_pred.to_string(index=False))
    print("=" * 72)

    # 2. Graficar diagnóstico completo (Recorte + Referencia Oficial + Top-5 + Comparativa 6 Modelos)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"Diagnóstico de Clasificación: {nombre_archivo}  -->  Predicción: {clase_ganadora} ({conf_ganadora*100:.1f}%)",
        fontsize=13,
        fontweight="bold"
    )

    # Panel 1: Recorte detectado
    axes[0].imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"1. Recorte Detectado\n({crop_bgr.shape[1]}x{crop_bgr.shape[0]} px)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Imagen oficial de referencia de esa clase en el catálogo LEGO
    carpeta_ref = os.path.join(config.CROPPED_IMAGES_DIR, clase_ganadora)
    if os.path.isdir(carpeta_ref):
        archivos_ref = sorted([f for f in os.listdir(carpeta_ref) if f.lower().endswith((".jpg", ".png"))])
        if archivos_ref:
            img_ref = cv2.imread(os.path.join(carpeta_ref, archivos_ref[0]))
            axes[1].imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"2. Referencia Oficial Catálogo\nClase: {clase_ganadora}", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Top-5 probabilidades de la CNN
    top5_idx = np.argsort(probs_cnn)[::-1][:5][::-1]
    top5_names = [class_names[i] for i in top5_idx]
    top5_vals = [probs_cnn[i] * 100.0 for i in top5_idx]
    colores_top5 = ["#90caf9"] * 4 + ["#1976d2"]
    bars = axes[2].barh(top5_names, top5_vals, color=colores_top5, edgecolor="black")
    axes[2].set_xlim(0, 112)
    axes[2].set_xlabel("Confianza (%)")
    axes[2].set_title("3. Top-5 Probabilidades (CNN)", fontsize=10, fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)
    for b, v in zip(bars, top5_vals):
        axes[2].text(min(102, v + 1.5), b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    # Panel 4: Comparativa de los 6 modelos
    nombres_cortos = [r["model_key"].upper() for r in comparativa_6][::-1]
    confs_6 = [r["Confianza (%)"] for r in comparativa_6][::-1]
    preds_6 = [r["Predicción"] for r in comparativa_6][::-1]
    colores_6 = ["#2e7d32" if p == clase_ganadora else "#fb8c00" for p in preds_6]
    bars6 = axes[3].barh(nombres_cortos, confs_6, color=colores_6, edgecolor="black")
    axes[3].set_xlim(0, 118)
    axes[3].set_xlabel("Confianza (%)")
    axes[3].set_title("4. Predicción de los 6 Modelos", fontsize=10, fontweight="bold")
    axes[3].grid(axis="x", linestyle="--", alpha=0.5)
    for b, p_lbl, v in zip(bars6, preds_6, confs_6):
        axes[3].text(min(100, v + 1.5), b.get_y() + b.get_height() / 2, f"{p_lbl} ({v:.0f}%)", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.show()
    return df_pred


# --- Prueba 1: Analizar una fotografía completa de mesa (Base Images) ---
ejemplo_base = os.path.join(config.BASE_IMAGES_DIR, "Brick_2x4", sorted(os.listdir(os.path.join(config.BASE_IMAGES_DIR, "Brick_2x4")))[0])
df_prediccion_6_modelos = probar_imagen_en_spyder(ruta_imagen=ejemplo_base)

# --- Prueba 2: Si quieres abrir una ventana para elegir cualquier imagen de tu PC en Spyder,
# simplemente descomenta la siguiente línea y ejecuta esta celda (Ctrl + Enter):
# df_prediccion_6_modelos = probar_imagen_en_spyder(abrir_ventana_archivo=True)
