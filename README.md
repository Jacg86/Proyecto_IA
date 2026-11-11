# Proyecto: Segmentación y Clasificación de Piezas LEGO mediante Machine Learning y Deep Learning

Sistema integral en Python para la identificación, localización mediante segmentación con OpenCV y clasificación multiclase de piezas LEGO en 20 categorías pertenecientes a las familias **Brick** y **Plate**.

---

## 1. Tabla de Contenidos
- [1. Descripción General](#1-descripción-general)
- [2. Estructura del Dataset](#2-estructura-del-dataset)
- [3. Resultados Experimentales y Comparativa de Modelos](#3-resultados-experimentales-y-comparativa-de-modelos)
- [4. Metodología por Etapas](#4-metodología-por-etapas)
  - [Etapa 1: Exploración y Calidad de Datos](#etapa-1-exploración-y-calidad-de-datos)
  - [Etapa 2: Particionado sin Fuga de Datos (Data Leakage)](#etapa-2-particionado-sin-fuga-de-datos-data-leakage)
  - [Etapa 3: Segmentación con OpenCV y Análisis de Watershed](#etapa-3-segmentación-con-opencv-y-análisis-de-watershed)
  - [Etapa 4: Extracción de Características (Forma vs. Forma+Color)](#etapa-4-extracción-de-características-forma-vs-formacolor)
  - [Etapa 5 y 6: Entrenamiento y Benchmarking de los 6 Modelos](#etapa-5-y-6-entrenamiento-y-benchmarking-de-los-6-modelos)
  - [Etapa 7: Pipeline de Inferencia End-to-End](#etapa-7-pipeline-de-inferencia-end-to-end)
- [5. Estructura del Repositorio](#5-estructura-del-repositorio)
- [6. Instrucciones de Ejecución](#6-instrucciones-de-ejecución)
  - [Ejecución en Spyder IDE (Recomendado)](#ejecución-en-spyder-ide-recomendado)
  - [Ejecución por Consola (CLI)](#ejecución-por-consola-cli)
  - [Cuaderno Jupyter](#cuaderno-jupyter)

---

## 2. Estructura del Dataset

El conjunto de datos comprende **4,580 imágenes recortadas** y **3,464 imágenes base** divididas en 20 clases:

- **Familia Brick (9 clases):** `Brick_1x1`, `Brick_1x2`, `Brick_1x3`, `Brick_1x4`, `Brick_2x2`, `Brick_2x2_L`, `Brick_2x2_Slope`, `Brick_2x3`, `Brick_2x4`.
- **Familia Plate (11 clases):** `Plate_1x1`, `Plate_1x1_Round`, `Plate_1x1_Slope`, `Plate_1x2`, `Plate_1x2_Grill`, `Plate_1x3`, `Plate_1x4`, `Plate_2x2`, `Plate_2x2_L`, `Plate_2x3`, `Plate_2x4`.

### Correspondencia y Prevención de Data Leakage
Cada imagen base (`2_Brick_2x2_L_180708213430.jpg`) posee una marca de tiempo idéntica con su respectivo recorte (`1_Brick_2x2_L_180708213430.jpg`). Para evitar que una misma fotografía aparezca simultáneamente en entrenamiento y prueba, el particionado se realiza agrupando por `timestamp_id`:
- **Entrenamiento (Train):** 3,206 muestras (70.0%)
- **Validación (Val):** 687 muestras (15.0%)
- **Prueba (Test):** 687 muestras (15.0%)

---

## 3. Resultados Experimentales y Comparativa de Modelos

Evaluación sobre el conjunto de prueba independiente (687 muestras balanceadas):

| Modelo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Tiempo de Entrenamiento | Latencia de Inferencia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN (Convolucional PyTorch)** | **95.63%** | **95.63%** | **95.58%** | **95.55%** | ~40.5 min | 11.94 ms/muestra |
| **SVM (Kernel RBF)** | **90.83%** | **91.46%** | **90.49%** | **90.77%** | **12.01 s** | **1.27 ms/muestra** |
| **Red Neuronal Artificial (ANN / MLP)** | **89.23%** | **89.19%** | **89.03%** | **88.99%** | **10.51 s** | **0.009 ms/muestra** |
| **Perceptrón (One-vs-Rest)** | 76.27% | 76.59% | 76.12% | 76.06% | 1.89 s | 0.026 ms/muestra |
| **SGAdaline (SGD MSE multiclase)** | 69.72% | 70.53% | 68.86% | 68.68% | 45.46 s | 0.030 ms/muestra |
| **Árbol de Decisión (CART)** | 67.83% | 68.11% | 67.02% | 67.02% | 1.48 s | 0.001 ms/muestra |

### Veredicto Técnico:
1. **Modelo Ganador Absoluto en Rendimiento:** **CNN (95.63% Exactitud)** debido a su capacidad para aprender automáticamente las invarianzas espaciales y las complejas configuraciones de tetones (studs) y proporciones 3D.
2. **Modelo Ganador en Eficiencia Computacional:** **SVM RBF (90.83% Exactitud)**, requiriendo solo 12 segundos de cómputo y 1.27 ms de latencia por muestra.

---

## 4. Metodología por Etapas

### Etapa 1: Exploración y Calidad de Datos
- Análisis automatizado en `preprocessing/dataset_explorer.py`.
- 0 imágenes corruptas y 0 duplicados exactos (MD5).
- Relación de aspecto promedio y distribución equilibrada entre categorías.

### Etapa 2: Preprocesamiento
- Estandarización de tamaño ($64 \times 64$) con interpolación adaptativa.
- Normalización min-max y estandarización `StandardScaler` ajustada únicamente en el conjunto de entrenamiento.

### Etapa 3: Segmentación con OpenCV y Análisis de Watershed
- Conversión a espacio HSV: umbralización Otsu sobre Canal S (piezas saturadas) + umbral inverso sobre Canal V ($V < 130$ para piezas oscuras/negras).
- Fusión binaria OR y operadores morfológicos (Apertura $5\times 5$ y Cierre $9\times 9$).
- Margen perimetral configurable ($\text{margin}=15$ px).
- Métricas frente a ground-truth: **Recall Píxel = 95.16%**, **IoU = 0.6457**, **Dice = 0.7011**.
- Evaluación de Watershed: Demostrado experimentalmente que separa piezas en contacto mediante transformada de distancia, pero requiere umbrales estrictos para no sobre-segmentar los relieves de los tetones.

### Etapa 4: Forma vs. Forma + Color
- **Solo Forma (301 dimensiones):** 6 métricas geométricas (área, perímetro, solidez, compacidad, aspecto, extensión), 7 Momentos invariantes de Hu con logaritmo con signo, y 288 descriptores HOG.
- **Forma + Color (325 dimensiones):** Las 301 anteriores + 24 histogramas HSV normalizados.
- **Hallazgo:** La forma geométrica representa el 88-90% de la capacidad predictiva; el color aporta una mejora marginal de +1.0% a +3.3%.

---

## 5. Estructura del Repositorio

```text
Proyecto_LEGO/
├── Proyecto_Final_Spyder.py      # Script principal por celdas (# %%) para Spyder IDE
├── config.py                     # Constantes, rutas, hiperparámetros y semillas
├── main.py                       # CLI maestro ejecutable del proyecto
├── pipeline.py                   # Pipeline integrado de inferencia end-to-end
├── generate_notebook.py          # Generador del cuaderno Jupyter
├── preprocessing/
│   ├── dataset_explorer.py       # Análisis exploratorio estadístico
│   └── data_loader.py            # Particionado estratificado sin data leakage
├── segmentation/
│   ├── hsv_segmenter.py          # Algoritmo de segmentación HSV Otsu y Watershed
│   ├── segmentation_evaluator.py # Cálculo de IoU, Dice, Precision y Recall
│   └── watershed_analysis.py     # Análisis experimental de Watershed en piezas en contacto
├── features/
│   └── feature_extractor.py      # Extracción de Hu, geometría, HOG e histogramas HSV
├── models/
│   ├── base_model.py             # Interfaz base estandarizada
│   ├── perceptron_model.py       # Perceptrón One-vs-Rest calibrado
│   ├── sgadaline_model.py        # SGAdaline con SGD y pérdida MSE
│   ├── svm_model.py              # Support Vector Machine con optimización de kernels
│   ├── decision_tree_model.py    # Árbol de decisión e importancia de características
│   ├── neural_net_model.py       # Red neuronal artificial MLP en PyTorch
│   ├── cnn_model.py              # Red neuronal convolucional en PyTorch
│   └── checkpoints/              # Pesos entrenados (.pth y .joblib)
├── evaluation/
│   ├── metrics.py                # Métricas cuantitativas y matriz de confusión
│   ├── visualizer.py             # Generación de gráficos comparativos
│   └── benchmark.py              # Orquestador del benchmark experimental
├── data/                         # Particiones JSON y matrices NPZ precalculadas
├── notebooks/
│   └── exploracion_y_modelado.ipynb  # Cuaderno interactivo
└── results/                      # Gráficas, matrices de confusión y reportes
    ├── exploration/
    ├── segmentation/
    ├── models/
    └── predictions/
```

---

## 6. Instrucciones de Ejecución

### Ejecución en Spyder IDE (Recomendado):
Abrir [`Proyecto_Final_Spyder.py`](Proyecto_Final_Spyder.py) en **Spyder 6** y presionar **`F5`** (o ejecutar bloque por bloque `# %%` con **`Ctrl + Enter`**):
- Todas las matrices estandarizadas (`X_train`, `X_test`, `y_train`, `y_test`), las matrices de confusión (`cm_cnn`, `cm_svm`, `cm_ann`, `cm_perceptron`, `cm_sgadaline`, `cm_arbol`) y la tabla comparativa (`df_comparacion_porcentaje`) quedan disponibles en el **Explorador de Variables (*Variable Explorer*)**.
- Todas las gráficas de desempeño, curvas de entrenamiento y pruebas de segmentación/clasificación se visualizan directamente en el panel **Plots**.
- En la última celda (`# %% 7`), se puede probar cualquier imagen ejecutando `probar_imagen_en_spyder(abrir_ventana_archivo=True)`.

### Ejecución por Consola (CLI):
```bash
# 1. Ejecutar el flujo completo de principio a fin:
python main.py --stage all

# 2. Ejecutar etapas individuales:
python main.py --stage explore      # Exploración del dataset
python main.py --stage segment      # Evaluación cuantitativa de segmentación OpenCV
python main.py --stage features     # Extracción y guardado de características
python main.py --stage train        # Entrenamiento y evaluación de los 6 modelos
python main.py --stage inference    # Inferencia demostrativa end-to-end sobre imágenes base
```

### Cuaderno Jupyter:
Abrir [`notebooks/exploracion_y_modelado.ipynb`](notebooks/exploracion_y_modelado.ipynb) en Jupyter Notebook o VS Code.

