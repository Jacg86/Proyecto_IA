import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Proyecto: Segmentación y Clasificación de Piezas LEGO con Machine Learning y Deep Learning\n",
    "**Dataset:** LEGO Bricks (20 categorías de las familias Brick y Plate)  \n",
    "**Modelos Evaluados:**\n",
    "1. Perceptrón (One-vs-Rest)\n",
    "2. SGAdaline (Stochastic Gradient Descent Adaline multiclase)\n",
    "3. Support Vector Machine (SVM con RBF, Polinómico y Lineal)\n",
    "4. Árboles de Decisión (CART con selección de hiperparámetros e importancia de características)\n",
    "5. Redes Neuronales Artificiales (ANN / MLP en PyTorch con Dropout y BatchNorm)\n",
    "6. Redes Neuronales Convolucionales (CNN en PyTorch con Data Augmentation)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "\n",
    "# Añadir raíz del proyecto al path\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "import config\n",
    "from preprocessing.dataset_explorer import explore_dataset\n",
    "from preprocessing.data_loader import create_or_load_splits\n",
    "from segmentation.hsv_segmenter import HSVSegmenter\n",
    "from features.feature_extractor import LEGOFeatureExtractor\n",
    "from pipeline import LEGODetectionPipeline\n",
    "\n",
    "print('PyTorch version:', torch.__version__)\n",
    "print('OpenCV version:', cv2.__version__)\n",
    "print('Dispositivo de cómputo:', config.DEVICE)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Exploración y Análisis del Dataset\n",
    "Inspección de las 4,580 imágenes recortadas y 3,464 imágenes base, verificación de integridad, detección de duplicados por MD5 y correspondencia por marca de tiempo."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "summary, crop_recs, base_recs = explore_dataset()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Segmentación con OpenCV en el Espacio HSV\n",
    "Segmentación combinando umbralización Otsu en el Canal S (piezas saturadas) y umbral inverso en Canal V (piezas oscuras), seguido de operaciones morfológicas y contornos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "segmenter = HSVSegmenter()\n",
    "sample_base_path = os.path.join(config.BASE_IMAGES_DIR, 'Brick_2x2_L', '2_Brick_2x2_L_180708213430.jpg')\n",
    "img_bgr = cv2.imread(sample_base_path)\n",
    "\n",
    "mask, pieces = segmenter.detect_and_extract_pieces(img_bgr, method='hsv_otsu')\n",
    "overlay = segmenter.draw_detections(img_bgr, pieces, labels=['Brick_2x2_L'], confidences=[0.98])\n",
    "\n",
    "fig, ax = plt.subplots(1, 3, figsize=(15, 5))\n",
    "ax[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))\n",
    "ax[0].set_title('1. Base Original')\n",
    "ax[0].axis('off')\n",
    "\n",
    "ax[1].imshow(mask, cmap='gray')\n",
    "ax[1].set_title('2. Máscara HSV Otsu')\n",
    "ax[1].axis('off')\n",
    "\n",
    "ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))\n",
    "ax[2].set_title('3. Detección y Contorno')\n",
    "ax[2].axis('off')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Extracción de Características: Forma vs. Forma + Color\n",
    "Carga de características precomputadas (301 dimensiones en Forma vs. 325 dimensiones en Forma + Color)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feat_shape = np.load('../data/features_shape_only.npz')\n",
    "feat_all = np.load('../data/features_shape_and_color.npz')\n",
    "\n",
    "print('Dimensión características Forma:', feat_shape['X_train'].shape[1])\n",
    "print('Dimensión características Forma + Color:', feat_all['X_train'].shape[1])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Tabla Comparativa de Rendimiento Experimental\n",
    "Resultados cuantitativos sobre el conjunto de prueba (687 muestras) en las mismas condiciones experimentales."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_results = pd.read_csv('../results/models/models_comparison_table.csv')\n",
    "display(df_results)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualizaciones Comparativas\n",
    "Gráficos de rendimiento global, tiempos de entrenamiento y latencia de inferencia."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_bench = cv2.imread('../results/models/models_benchmark_comparison.png')\n",
    "plt.figure(figsize=(16, 5))\n",
    "plt.imshow(cv2.cvtColor(img_bench, cv2.COLOR_BGR2RGB))\n",
    "plt.axis('off')\n",
    "plt.title('Comparación de Modelos: Accuracy, F1-Score, Train Time y Latencia', fontsize=13, fontweight='bold')\n",
    "plt.show()\n",
    "\n",
    "img_scen = cv2.imread('../results/models/shape_vs_color_comparison.png')\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.imshow(cv2.cvtColor(img_scen, cv2.COLOR_BGR2RGB))\n",
    "plt.axis('off')\n",
    "plt.title('Impacto de Color vs. Solo Forma en Modelos Tradicionales', fontsize=13, fontweight='bold')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Pipeline End-to-End: Segmentación y Clasificación\n",
    "Ejemplo de inferencia recibiendo una fotografía original, aislando la pieza y clasificándola mediante la red neuronal convolucional entrenada."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = LEGODetectionPipeline(model_type='cnn')\n",
    "sample_test_img = os.path.join(config.BASE_IMAGES_DIR, 'Plate_1x1', '2_Plate_1x1_180722193025.jpg')\n",
    "res = pipeline.process_image(sample_test_img, segmentation_method='hsv_otsu', save_output=False)\n",
    "\n",
    "plt.figure(figsize=(9, 6))\n",
    "plt.imshow(cv2.cvtColor(res['overlay'], cv2.COLOR_BGR2RGB))\n",
    "plt.title(f\"Detección: {res['pieces'][0]['class_name']} (Confianza: {res['pieces'][0]['confidence']*100:.1f}%)\", fontsize=12, fontweight='bold')\n",
    "plt.axis('off')\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

out_path = r"c:\Users\Asus\Desktop\Programas\IA2\Proyecto_LEGO\notebooks\exploracion_y_modelado.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print("Jupyter Notebook generado exitosamente en:", out_path)
