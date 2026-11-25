"""
Interfaz web en Streamlit para el proyecto de segmentación y clasificación de piezas LEGO.
Reutiliza directamente el pipeline, la segmentación OpenCV, la extracción de características
y los modelos entrenados existentes en el repositorio.
"""
import csv
import io
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_DISPLAY_NAMES = {
    "cnn": "CNN (Convolucional PyTorch - 95.63% Acc)",
    "svm": "SVM Kernel RBF (90.83% Acc)",
    "ann": "Red Neuronal Artificial MLP (89.23% Acc)",
    "perceptron": "Perceptrón One-vs-Rest (76.27% Acc)",
    "sgadaline": "SGAdaline SGD MSE (69.72% Acc)",
    "decision_tree": "Árbol de Decisión CART (67.83% Acc)",
}

MODEL_CHECKPOINT_FILES = {
    "cnn": "cnn_model.pth",
    "svm": "svm_model.joblib",
    "ann": "ann_model.pth",
    "perceptron": "perceptron_model.joblib",
    "sgadaline": "sgadaline_model.joblib",
    "decision_tree": "decision_tree_model.joblib",
}

SEGMENTATION_METHODS = {
    "hsv_otsu": "HSV + Otsu (Estándar del proyecto)",
    "hsv_adaptive": "HSV Adaptativo (Piezas claras/blancas)",
    "universal_adaptive": "Adaptativo Universal (Fondos externos/iluminación variable)",
    "bg_sub": "Sustracción de fondo (Mesa retroiluminada)",
    "direct_crop": "Recorte directo (Imagen ya recortada)",
}

MODEL_CSV_NAMES = {
    "cnn": "CNN",
    "svm": "SVM (rbf)",
    "ann": "Red Neuronal (ANN)",
    "perceptron": "Perceptrón",
    "sgadaline": "SGAdaline",
    "decision_tree": "Árbol de Decisión",
}


st.set_page_config(
    page_title="LEGO Vision - Detector y Clasificador de Piezas",
    page_icon="🧱",
    layout="wide",
)


@st.cache_data
def load_benchmark_metrics():
    """Carga las métricas experimentales reales desde results/models/models_comparison_table.csv."""
    csv_path = os.path.join(PROJECT_ROOT, "results", "models", "models_comparison_table.csv")
    metrics = {}
    if not os.path.exists(csv_path):
        return metrics
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Model", "").strip()
                if name:
                    metrics[name] = {
                        "accuracy": float(row["Accuracy"]) * 100.0,
                        "f1_macro": float(row["F1-Score (Macro)"]) * 100.0,
                        "latency_ms": float(row["Inference Latency (ms)"]),
                    }
    except Exception:
        return {}
    return metrics


@st.cache_resource
def get_pipeline(model_type: str):
    """Inicializa y almacena en caché el pipeline existente sin reentrenar modelos."""
    from pipeline import LEGODetectionPipeline

    return LEGODetectionPipeline(model_type=model_type)


def get_checkpoint_path(model_type: str) -> str:
    filename = MODEL_CHECKPOINT_FILES.get(model_type, "")
    return os.path.join(PROJECT_ROOT, "models", "checkpoints", filename)


def is_model_ready(pipeline, model_type: str) -> tuple[bool, str]:
    """Verifica que el archivo de pesos exista y que el clasificador se haya cargado."""
    ckpt_path = get_checkpoint_path(model_type)
    if not os.path.exists(ckpt_path):
        return False, f"No se encontró el archivo del modelo esperado en `{ckpt_path}`."

    clf = pipeline._get_or_load_model(model_type)
    if clf is None:
        return False, f"No fue posible cargar el modelo `{model_type}` desde `{ckpt_path}`."

    if model_type in ("cnn", "ann") and getattr(clf, "net", None) is None:
        return False, f"La arquitectura PyTorch del modelo `{model_type}` no se inicializó correctamente."

    return True, ckpt_path


def classifier_provides_probabilities(clf, model_type: str) -> bool:
    """Comprueba si el clasificador cargado expone un método real de probabilidades."""
    if clf is None:
        return False
    if model_type == "cnn":
        return callable(getattr(clf, "predict_proba_single_image", None)) and getattr(clf, "net", None) is not None
    return callable(getattr(clf, "predict_proba", None))


def decode_uploaded_image(uploaded_file) -> tuple[np.ndarray, Image.Image]:
    """Valida el formato y decodifica el archivo subido a RGB (PIL) y BGR (OpenCV)."""
    filename = getattr(uploaded_file, "name", "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Formato de archivo no compatible (`{ext or 'desconocido'}`). "
            "Solo se admiten imágenes JPG, JPEG o PNG."
        )

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("El archivo cargado está vacío.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img_check:
            img_check.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img_pil:
            pil_rgb = ImageOps.exif_transpose(img_pil).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"No se pudo decodificar la imagen cargada: {exc}") from exc

    img_rgb = np.array(pil_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr, pil_rgb


def run_analysis(
    pipeline,
    img_bgr: np.ndarray,
    model_type: str,
    segmentation_method: str,
    min_confidence: float,
    use_watershed: bool,
    filter_skin: bool,
    only_main_piece: bool,
    normalize_bg: bool,
    use_tta: bool,
) -> dict:
    """
    Ejecuta la segmentación OpenCV, extracción de recortes y clasificación existente,
    registrando el tiempo de procesamiento y validando detecciones y probabilidades.
    """
    import config

    start_time = time.perf_counter()

    # 1. Segmentación y extracción de máscaras, contornos y recortes mediante OpenCV (HSVSegmenter)
    binary_mask, pieces = pipeline.segmenter.detect_and_extract_pieces(
        img_bgr,
        method=segmentation_method,
        use_watershed=use_watershed,
        filter_skin=filter_skin,
        only_main_piece=only_main_piece,
    )

    if not pieces:
        elapsed = time.perf_counter() - start_time
        return {
            "total_detected": 0,
            "pieces": [],
            "mask": binary_mask,
            "overlay": None,
            "processing_time": elapsed,
            "has_probabilities": False,
        }

    clf = pipeline._get_or_load_model(model_type)
    has_probs = classifier_provides_probabilities(clf, model_type)

    detected_pieces = []
    labels = []
    confidences_for_overlay = []

    # 2. Clasificación de cada recorte reutilizando el preprocesamiento y modelo del pipeline
    for idx, piece in enumerate(pieces):
        crop_bgr = piece["crop"]
        if crop_bgr is None or crop_bgr.size == 0:
            continue

        if normalize_bg:
            crop_bgr = pipeline.segmenter.normalize_crop_background(crop_bgr, piece.get("piece_mask"))

        if has_probs:
            class_idx, conf, raw_class_name, probs = pipeline.predict_crop_with_model(
                crop_bgr, model_type=model_type, use_tta=use_tta
            )
            if probs is None or len(probs) != config.NUM_CLASSES:
                conf_val = None
            else:
                conf_val = float(conf)
        else:
            # Respaldo seguro en caso de un clasificador sin probabilidades: no inventar confianza
            feats_scaled = pipeline.extractor.extract_and_scale_single(crop_bgr, scenario="shape_and_color")
            class_idx = int(clf.predict(feats_scaled)[0])
            raw_class_name = config.IDX_TO_CLASS[class_idx]
            conf_val = None

        if conf_val is not None and conf_val < min_confidence:
            display_label = f"Desconocida ({raw_class_name}?)"
            is_confident = False
        else:
            display_label = raw_class_name
            is_confident = True

        labels.append(display_label)
        if conf_val is not None:
            confidences_for_overlay.append(conf_val)

        x, y, w, h = piece["tight_bbox"]
        detected_pieces.append(
            {
                "piece_id": idx + 1,
                "objeto": f"Pieza {idx + 1}",
                "class_name": display_label,
                "base_class": raw_class_name,
                "confidence": conf_val,
                "is_confident": is_confident,
                "tight_bbox": (int(x), int(y), int(w), int(h)),
                "location_str": f"x={int(x)}, y={int(y)}, ancho={int(w)}, alto={int(h)}",
                "contour_area": float(piece["area"]),
                "crop_rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
            }
        )

    if not detected_pieces:
        elapsed = time.perf_counter() - start_time
        return {
            "total_detected": 0,
            "pieces": [],
            "mask": binary_mask,
            "overlay": None,
            "processing_time": elapsed,
            "has_probabilities": has_probs,
        }

    # 3. Dibujo de contornos, bounding boxes y etiquetas con la función existente
    overlay_confidences = (
        confidences_for_overlay if len(confidences_for_overlay) == len(detected_pieces) else None
    )
    annotated_bgr = pipeline.segmenter.draw_detections(
        img_bgr,
        pieces[: len(detected_pieces)],
        labels=labels,
        confidences=overlay_confidences,
    )

    elapsed = time.perf_counter() - start_time
    return {
        "total_detected": len(detected_pieces),
        "pieces": detected_pieces,
        "mask": binary_mask,
        "overlay": annotated_bgr,
        "processing_time": elapsed,
        "has_probabilities": overlay_confidences is not None,
    }


def encode_png_bytes(img_bgr: np.ndarray) -> bytes:
    """Codifica una imagen BGR de OpenCV a PNG en memoria para descarga."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen procesada en formato PNG.")
    return buf.tobytes()


def main():
    st.title("LEGO Vision - Detector y Clasificador de Piezas")
    st.markdown(
        "Sistema de visión artificial para la segmentación, detección y clasificación automática de piezas LEGO."
    )

    benchmark_metrics = load_benchmark_metrics()

    # --- Barra lateral ---
    with st.sidebar:
        st.header("Información del Proyecto")
        st.markdown(
            """
            Esta aplicación integra el pipeline implementado en el proyecto:
            - **Segmentación:** OpenCV en espacio de color HSV, umbralización y morfología (`HSVSegmenter`).
            - **Extracción y preprocesamiento:** Redimensionado a `64×64` px / descriptores de forma, Hu, HOG e histogramas HSV (`LEGOFeatureExtractor`).
            - **Clasificación:** 20 categorías LEGO (9 clases **Brick** y 11 clases **Plate**).
            """
        )

        st.divider()
        st.header("Configuración del Modelo")

        model_options = list(MODEL_DISPLAY_NAMES.keys())
        selected_model = st.selectbox(
            "Modelo de clasificación",
            options=model_options,
            index=0,  # 'cnn' es el mejor modelo experimental
            format_func=lambda k: f"⭐ {MODEL_DISPLAY_NAMES[k]} (Mejor modelo)" if k == "cnn" else MODEL_DISPLAY_NAMES[k],
            help="Por defecto se utiliza la CNN PyTorch, seleccionada como mejor modelo en la evaluación experimental.",
        )

        csv_key = MODEL_CSV_NAMES.get(selected_model)
        if csv_key and csv_key in benchmark_metrics:
            m_info = benchmark_metrics[csv_key]
            st.caption(
                f"Desempeño en test: **Accuracy {m_info['accuracy']:.2f}%** | "
                f"**F1-Macro {m_info['f1_macro']:.2f}%** | "
                f"Latencia {m_info['latency_ms']:.3f} ms"
            )

        st.subheader("Opciones de segmentación e inferencia")
        segmentation_method = st.selectbox(
            "Método de segmentación OpenCV",
            options=list(SEGMENTATION_METHODS.keys()),
            index=0,
            format_func=lambda k: SEGMENTATION_METHODS[k],
        )

        min_confidence = st.slider(
            "Umbral mínimo de confianza",
            min_value=0.0,
            max_value=0.95,
            value=0.55,
            step=0.05,
            help="Si la confianza es inferior a este umbral, la pieza se marca como 'Desconocida (clase?)' igual que en pipeline.py.",
        )

        with st.expander("Ajustes avanzados del pipeline"):
            use_watershed = st.checkbox(
                "Separar piezas en contacto (Watershed)",
                value=False,
                help="Aplica apply_watershed_separation para separar piezas que se tocan.",
            )
            filter_skin = st.checkbox(
                "Filtrar tonos de piel (YCrCb)",
                value=False,
                help="Útil si la pieza se sostiene con la mano.",
            )
            only_main_piece = st.checkbox(
                "Detectar solo la pieza principal",
                value=False,
                help="Conserva únicamente el contorno de mayor área.",
            )
            normalize_bg = st.checkbox(
                "Normalizar fondo del recorte",
                value=False,
                help="Reemplaza el fondo externo por el fondo retroiluminado de referencia.",
            )
            use_tta = st.checkbox(
                "Usar Test-Time Augmentation (TTA)",
                value=False,
                help="Evalúa espejo horizontal y rotación de 90° en piezas verticales con baja confianza.",
            )

        ckpt_path = get_checkpoint_path(selected_model)
        if os.path.exists(ckpt_path):
            st.success(f"Checkpoint disponible: `{os.path.basename(ckpt_path)}`")
        else:
            st.error(f"Checkpoint no encontrado: `{ckpt_path}`")

    # --- Carga de fotografía ---
    st.subheader("Cargar fotografía")
    uploaded_file = st.file_uploader(
        "Selecciona una fotografía con una o varias piezas LEGO (JPG, JPEG o PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Sube una imagen en formato JPG, JPEG o PNG para iniciar el análisis.")
        return

    try:
        img_bgr, pil_rgb = decode_uploaded_image(uploaded_file)
    except ValueError as err:
        st.error(f"Archivo inválido: {err}")
        return

    file_signature = (
        uploaded_file.name,
        uploaded_file.size,
        selected_model,
        segmentation_method,
        min_confidence,
        use_watershed,
        filter_skin,
        only_main_piece,
        normalize_bg,
        use_tta,
    )

    if st.session_state.get("last_signature") != file_signature:
        st.session_state["analysis_result"] = None

    col_preview, col_action = st.columns([2, 1])
    with col_preview:
        st.image(pil_rgb, caption=f"Vista previa: {uploaded_file.name} ({pil_rgb.width}×{pil_rgb.height} px)", use_container_width=True)
    with col_action:
        st.markdown("### Ejecución")
        st.write(f"**Archivo:** `{uploaded_file.name}`")
        st.write(f"**Dimensiones:** {pil_rgb.width} × {pil_rgb.height} px")
        st.write(f"**Modelo:** {MODEL_DISPLAY_NAMES[selected_model]}")
        analyze_clicked = st.button("Analizar imagen", type="primary", use_container_width=True)

    if analyze_clicked:
        with st.spinner("Cargando modelo y analizando la imagen..."):
            try:
                pipeline = get_pipeline(selected_model)
                ready, ready_msg = is_model_ready(pipeline, selected_model)
                if not ready:
                    st.error(f"Modelo no disponible: {ready_msg}")
                    return

                result = run_analysis(
                    pipeline=pipeline,
                    img_bgr=img_bgr,
                    model_type=selected_model,
                    segmentation_method=segmentation_method,
                    min_confidence=min_confidence,
                    use_watershed=use_watershed,
                    filter_skin=filter_skin,
                    only_main_piece=only_main_piece,
                    normalize_bg=normalize_bg,
                    use_tta=use_tta,
                )
                st.session_state["analysis_result"] = result
                st.session_state["last_signature"] = file_signature
                st.session_state["analyzed_filename"] = uploaded_file.name
                st.session_state["analyzed_model"] = selected_model
            except Exception as exc:
                st.error(f"Ocurrió un error durante el procesamiento de la imagen: {exc}")
                return

    result = st.session_state.get("analysis_result")
    if result is None:
        return

    st.divider()

    # Validación: fotografía sin piezas detectables / predicciones vacías
    if result["total_detected"] == 0 or not result["pieces"]:
        st.warning(
            "No se detectaron piezas LEGO en la fotografía con la configuración actual. "
            "Si la imagen tiene fondo complejo o iluminación distinta al conjunto original, "
            "prueba cambiar el método de segmentación en la barra lateral (por ejemplo, "
            "**Adaptativo Universal**, **HSV Adaptativo** o **Recorte directo**)."
        )
        tab_orig, tab_mask = st.tabs(["1. Imagen original", "2. Máscara de segmentación"])
        with tab_orig:
            st.image(pil_rgb, caption="Imagen original cargada", use_container_width=True)
        with tab_mask:
            st.image(result["mask"], caption="Máscara binaria generada por OpenCV (sin contornos válidos)", use_container_width=True)
        return

    # --- Resumen de análisis ---
    st.subheader("Resumen de análisis")
    unique_categories = sorted({p["base_class"] for p in result["pieces"]})
    used_model_key = st.session_state.get("analyzed_model", selected_model)
    used_model_label = MODEL_CSV_NAMES.get(used_model_key, used_model_key.upper())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Piezas detectadas", result["total_detected"])
    m2.metric("Categorías diferentes", len(unique_categories))
    m3.metric("Modelo utilizado", used_model_label)
    m4.metric("Tiempo de procesamiento", f"{result['processing_time']:.3f} s")

    # --- Visualización de resultados en pestañas ---
    st.subheader("Visualización de resultados")
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Imagen original",
            "2. Máscara de segmentación",
            "3. Imagen procesada",
            "4. Piezas individuales",
        ]
    )

    with tab1:
        st.image(pil_rgb, caption="Fotografía original cargada por el usuario", use_container_width=True)

    with tab2:
        st.image(
            result["mask"],
            caption="Máscara binaria generada por OpenCV",
            use_container_width=True,
        )

    with tab3:
        overlay_rgb = cv2.cvtColor(result["overlay"], cv2.COLOR_BGR2RGB)
        st.image(
            overlay_rgb,
            caption="Contornos, bounding boxes y clasificación predicha",
            use_container_width=True,
        )

        # --- Descarga de resultados ---
        try:
            png_bytes = encode_png_bytes(result["overlay"])
            base_name = os.path.splitext(st.session_state.get("analyzed_filename", "resultado"))[0]
            st.download_button(
                label="Descargar imagen procesada",
                data=png_bytes,
                file_name=f"{base_name}_procesada.png",
                mime="image/png",
                type="primary",
            )
        except Exception as exc:
            st.error(f"No se pudo preparar la descarga de la imagen procesada: {exc}")

    with tab4:
        cols_per_row = 3
        pieces_list = result["pieces"]
        for start_idx in range(0, len(pieces_list), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for col_offset, piece in enumerate(pieces_list[start_idx : start_idx + cols_per_row]):
                with row_cols[col_offset]:
                    conf_text = (
                        f"{piece['confidence'] * 100:.2f}%"
                        if piece["confidence"] is not None
                        else "No disponible"
                    )
                    st.image(
                        piece["crop_rgb"],
                        caption=f"{piece['objeto']} — {piece['class_name']}",
                        use_container_width=True,
                    )
                    st.markdown(
                        f"- **Categoría predicha:** `{piece['class_name']}`\n"
                        f"- **Confianza:** {conf_text}\n"
                        f"- **Ubicación:** {piece['location_str']}"
                    )

    # --- Tabla de resultados ---
    st.subheader("Tabla de resultados")
    table_rows = []
    for piece in result["pieces"]:
        conf_str = (
            f"{piece['confidence'] * 100:.2f}%"
            if piece["confidence"] is not None
            else "No disponible"
        )
        table_rows.append(
            {
                "Objeto": piece["objeto"],
                "Categoría predicha": piece["class_name"],
                "Confianza": conf_str,
                "Ubicación": piece["location_str"],
            }
        )

    df_results = pd.DataFrame(table_rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
