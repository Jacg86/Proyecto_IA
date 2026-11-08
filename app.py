"""
Interactive Streamlit Web Dashboard for LEGO Brick Segmentation and Classification.
Provides interactive inspection of OpenCV segmentation parameters, model inference,
and visual benchmark comparison.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from segmentation.hsv_segmenter import HSVSegmenter
from pipeline import LEGODetectionPipeline

st.set_page_config(
    page_title="LEGO Vision AI: Segmentación y Clasificación",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e53935;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧱 LEGO Vision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema de Segmentación OpenCV y Clasificación Multiclase con Machine Learning & Deep Learning</div>', unsafe_allow_html=True)

# Sidebar Navigation
mode = st.sidebar.radio(
    "Navegación:",
    ["🔍 Detección e Inferencia End-to-End", "📊 Benchmark y Métricas de los 6 Modelos", "🧪 Análisis de Segmentación & Watershed", "📂 Explorador del Dataset"]
)

# Initialize segmentation and pipeline
@st.cache_resource
def load_cached_pipeline(model_type="cnn"):
    return LEGODetectionPipeline(model_type=model_type)

def get_pipeline(model_type="cnn"):
    pipeline = load_cached_pipeline(model_type=model_type)
    # Always refresh segmenter instance to avoid cached method signature mismatches
    pipeline.segmenter = HSVSegmenter()
    return pipeline

if mode == "🔍 Detección e Inferencia End-to-End":
    st.subheader("Detección y Clasificación en Tiempo Real")
    
    col_input, col_settings = st.columns([1, 1])
    
    with col_settings:
        st.markdown("#### Configuración de Segmentación (OpenCV)")
        seg_method = st.selectbox(
            "Método de Segmentación:",
            ["hsv_otsu", "universal_adaptive", "hsv_adaptive", "bg_sub"],
            format_func=lambda x: "HSV Dual Otsu (Fondo Retroiluminado)" if x=="hsv_otsu" else (
                "Universal Adaptativo (Fotos Externas / Papel / Mesa)" if x=="universal_adaptive" else (
                    "HSV Adaptativo + Gradiente" if x=="hsv_adaptive" else "Sustracción de Fondo"
                )
            )
        )
        margin = st.slider("Margen de recorte (píxeles):", min_value=5, max_value=30, value=15, step=1)
        min_area = st.slider("Área mínima de contorno (px):", min_value=100, max_value=2000, value=400, step=50)
        use_watershed = st.checkbox("Aplicar Separación Watershed (piezas en contacto)", value=False)
        filter_skin = st.checkbox("Filtrar piel humana (elimina dedos/manos que sostienen la pieza)", value=False)
        min_conf = st.slider("Umbral de Confianza Mínima (%):", min_value=30, max_value=90, value=55, step=5) / 100.0
        
        st.markdown("#### Configuración de Clasificador")
        model_choice = st.selectbox("Modelo para Inferencia:", ["cnn", "svm"], format_func=lambda x: "Red Neuronal Convolucional (CNN - 95.6% Acc)" if x=="cnn" else "Support Vector Machine (SVM RBF - 90.8% Acc)")
        
    with col_input:
        st.markdown("#### Selección de Imagen de Entrada")
        input_source = st.radio("Fuente de imagen:", ["Muestras del Dataset Base", "Subir Imagen Propia"], horizontal=True)
        
        selected_img_bgr = None
        sample_name = ""
        
        if input_source == "Muestras del Dataset Base":
            # List available classes in Base Images
            base_classes = sorted([d for d in os.listdir(config.BASE_IMAGES_DIR) if os.path.isdir(os.path.join(config.BASE_IMAGES_DIR, d))])
            chosen_cls = st.selectbox("Familia / Categoría:", base_classes)
            cls_path = os.path.join(config.BASE_IMAGES_DIR, chosen_cls)
            available_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))]
            chosen_file = st.selectbox("Archivo de imagen base:", available_files)
            
            full_path = os.path.join(cls_path, chosen_file)
            selected_img_bgr = cv2.imread(full_path)
            sample_name = chosen_file
        else:
            uploaded_file = st.file_uploader("Arrastre o seleccione una fotografía de piezas LEGO:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                selected_img_bgr = cv2.imdecode(file_bytes, 1)
                sample_name = uploaded_file.name

    if selected_img_bgr is not None:
        st.markdown("---")
        st.markdown(f"### Resultados del Pipeline: `{sample_name}`")
        
        pipeline = get_pipeline(model_type=model_choice)
        pipeline.segmenter.margin = margin
        pipeline.segmenter.min_area = min_area
        
        # Execute segmentation
        mask, pieces = pipeline.segmenter.detect_and_extract_pieces(
            selected_img_bgr, method=seg_method, use_watershed=use_watershed, filter_skin=filter_skin
        )
        
        labels = []
        confidences = []
        for p in pieces:
            if pipeline.model_type == "cnn" and pipeline.classifier.net is not None:
                c_idx, conf, c_name = pipeline.classifier.predict_single_image(p["crop"])
            elif pipeline.model_type == "svm" and pipeline.classifier is not None:
                from features.feature_extractor import LEGOFeatureExtractor
                extractor = LEGOFeatureExtractor()
                feats = extractor.extract_all_features(p["crop"]).reshape(1, -1)
                probs = pipeline.classifier.predict_proba(feats)[0]
                c_idx = int(np.argmax(probs))
                conf = float(probs[c_idx])
                c_name = config.IDX_TO_CLASS[c_idx]
            else:
                c_name = "LEGO Piece"
                conf = 0.50
                
            if conf < min_conf:
                display_lbl = f"Dudoso ({c_name}?)"
            else:
                display_lbl = c_name
                
            labels.append(display_lbl)
            confidences.append(conf)
            
        overlay = pipeline.segmenter.draw_detections(selected_img_bgr, pieces, labels, confidences)
        masked_img = cv2.bitwise_and(selected_img_bgr, selected_img_bgr, mask=mask)
        
        # Display 4-panel view
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.image(cv2.cvtColor(selected_img_bgr, cv2.COLOR_BGR2RGB), caption="1. Fotografía Original", use_container_width=True)
        with p2:
            st.image(mask, caption="2. Máscara OpenCV", use_container_width=True)
        with p3:
            st.image(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), caption="3. Pieza Aislada", use_container_width=True)
        with p4:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="4. Detección y Clasificación", use_container_width=True)
            
        # Display detection summary cards
        st.markdown(f"**Total piezas detectadas:** `{len(pieces)}`")
        if pieces:
            cols = st.columns(min(4, len(pieces)))
            for idx, (piece, lbl, conf) in enumerate(zip(pieces, labels, confidences)):
                with cols[idx % len(cols)]:
                    st.info(f"**Pieza #{idx+1}:** {lbl}\n\n**Confianza:** {conf*100:.1f}%\n\n**Área:** {piece['area']:.0f} px")
                    st.image(cv2.cvtColor(piece["crop"], cv2.COLOR_BGR2RGB), caption=f"Recorte #{idx+1}", width=120)

elif mode == "📊 Benchmark y Métricas de los 6 Modelos":
    st.subheader("Evaluación Experimental Comparativa de los 6 Modelos")
    
    csv_path = os.path.join(config.RESULTS_DIR, "models", "models_comparison_table.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Format columns for display
        df_display = df.copy()
        df_display["Accuracy"] = df_display["Accuracy"].map(lambda x: f"{x*100:.2f}%")
        df_display["F1-Score (Macro)"] = df_display["F1-Score (Macro)"].map(lambda x: f"{x*100:.2f}%")
        df_display["Precision (Macro)"] = df_display["Precision (Macro)"].map(lambda x: f"{x*100:.2f}%")
        df_display["Recall (Macro)"] = df_display["Recall (Macro)"].map(lambda x: f"{x*100:.2f}%")
        df_display["Train Time (s)"] = df_display["Train Time (s)"].map(lambda x: f"{x:.2f}s")
        df_display["Inference Latency (ms)"] = df_display["Inference Latency (ms)"].map(lambda x: f"{x:.3f} ms")
        
        st.dataframe(df_display, use_container_width=True)
        
        # Visual Comparisons
        st.markdown("### Gráficas Comparativas")
        c1, c2 = st.columns(2)
        bench_img = os.path.join(config.RESULTS_DIR, "models", "models_benchmark_comparison.png")
        scen_img = os.path.join(config.RESULTS_DIR, "models", "shape_vs_color_comparison.png")
        if os.path.exists(bench_img):
            c1.image(bench_img, caption="Comparativa de Rendimiento, Tiempo de Entrenamiento y Latencia", use_container_width=True)
        if os.path.exists(scen_img):
            c2.image(scen_img, caption="Impacto de Características: Solo Forma vs. Forma + Color", use_container_width=True)
            
        st.markdown("### Curvas de Aprendizaje y Matrices de Confusión")
        t1, t2, t3, t4 = st.tabs(["CNN (Convolucional)", "SVM (RBF)", "ANN (MLP)", "Árbol de Decisión"])
        with t1:
            k1, k2 = st.columns(2)
            k1.image(os.path.join(config.RESULTS_DIR, "models", "cnn_learning_curves.png"), caption="Curvas de Aprendizaje CNN", use_container_width=True)
            k2.image(os.path.join(config.RESULTS_DIR, "models", "cm_cnn.png"), caption="Matriz de Confusión CNN", use_container_width=True)
        with t2:
            st.image(os.path.join(config.RESULTS_DIR, "models", "cm_svm.png"), caption="Matriz de Confusión SVM RBF", width=700)
        with t3:
            k1, k2 = st.columns(2)
            k1.image(os.path.join(config.RESULTS_DIR, "models", "ann_learning_curves.png"), caption="Curvas de Aprendizaje ANN", use_container_width=True)
            k2.image(os.path.join(config.RESULTS_DIR, "models", "cm_ann.png"), caption="Matriz de Confusión ANN", use_container_width=True)
        with t4:
            k1, k2 = st.columns(2)
            k1.image(os.path.join(config.RESULTS_DIR, "models", "decision_tree_feature_importance.png"), caption="Importancia de Características", use_container_width=True)
            k2.image(os.path.join(config.RESULTS_DIR, "models", "cm_decision_tree.png"), caption="Matriz de Confusión Árbol", use_container_width=True)

elif mode == "🧪 Análisis de Segmentación & Watershed":
    st.subheader("Evaluación de Segmentación OpenCV y Algoritmo Watershed")
    
    ws_img_path = os.path.join(config.RESULTS_DIR, "segmentation", "watershed_separation_analysis.png")
    if os.path.exists(ws_img_path):
        st.image(ws_img_path, caption="Análisis Experimental de Watershed en Piezas en Contacto", use_container_width=True)
        
    st.markdown("""
    #### Conclusiones Técnicas de la Segmentación:
    1. **HSV Dual Otsu:** Combina de forma óptima el canal S (piezas saturadas) y el canal V inverso (piezas negras/oscuras), logrando un **Recall de 95.16%** y **IoU de 0.6457** frente a máscaras de referencia.
    2. **Operaciones Morfológicas:** Un kernel de apertura ($5\\times 5$) elimina reflejos espurios y un cierre ($9\\times 9$) sella los huecos internos causados por sombras en los tetones superiores.
    3. **Evaluación de Watershed:** Cuando dos piezas entran en contacto directo, Watershed logra separar las componentes utilizando la transformada de distancia. No obstante, en piezas individuales su aplicación indiscriminada puede provocar sobre-segmentación en los relieves de los tetones.
    """)

elif mode == "📂 Explorador del Dataset":
    st.subheader("Composición y Estadísticas del Dataset")
    dist_img = os.path.join(config.RESULTS_DIR, "exploration", "class_distribution.png")
    res_img = os.path.join(config.RESULTS_DIR, "exploration", "resolution_and_aspect_ratio.png")
    
    if os.path.exists(dist_img):
        st.image(dist_img, caption="Distribución de Clases (4,580 muestras en 20 categorías)", use_container_width=True)
    if os.path.exists(res_img):
        st.image(res_img, caption="Distribución de Dimensiones y Relaciones de Aspecto", use_container_width=True)

