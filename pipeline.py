"""
End-to-End Pipeline module.
Integrates OpenCV HSV segmentation with the 6 trained classifiers (CNN, SVM, ANN,
Perceptron, SGAdaline, Decision Tree) and standardized feature extraction.
Takes a photograph, segments all LEGO pieces, crops them, classifies each piece,
and generates rich visual overlays with labels and confidence scores.
"""
import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from segmentation.hsv_segmenter import HSVSegmenter
from features.feature_extractor import LEGOFeatureExtractor
from models.cnn_model import LEGOCNN, LEGOConvNet
from models.svm_model import LEGOSVM
from models.neural_net_model import LEGONeuralNetwork, MLPNet
from models.perceptron_model import LEGOPerceptron
from models.sgadaline_model import MulticlassSGAdaline
from models.decision_tree_model import LEGODecisionTree

MODEL_DISPLAY_NAMES = {
    "cnn": "CNN (Convolucional PyTorch - 95.63% Acc)",
    "svm": "SVM Kernel RBF (90.83% Acc)",
    "ann": "Red Neuronal Artificial MLP (89.23% Acc)",
    "perceptron": "Perceptrón One-vs-Rest (76.27% Acc)",
    "sgadaline": "SGAdaline SGD MSE (69.72% Acc)",
    "decision_tree": "Árbol de Decisión CART (67.83% Acc)",
}

class LEGODetectionPipeline:
    def __init__(self, model_type="cnn"):
        self.model_type = model_type
        self.segmenter = HSVSegmenter(
            min_area=config.MIN_CONTOUR_AREA,
            margin=config.SEGMENTATION_MARGIN
        )
        self.extractor = LEGOFeatureExtractor()
        self.extractor.load_or_fit_scaler(scenario="shape_and_color")
        self.models_cache = {}
        self.classifier = self._get_or_load_model(self.model_type)

    def _get_or_load_model(self, model_type):
        if model_type in self.models_cache:
            return self.models_cache[model_type]

        checkpoints_dir = os.path.join(config.MODELS_DIR, "checkpoints")
        model = None

        if model_type == "cnn":
            weights_path = os.path.join(checkpoints_dir, "cnn_model.pth")
            model = LEGOCNN(num_classes=config.NUM_CLASSES)
            if os.path.exists(weights_path):
                model.net = LEGOConvNet(num_classes=config.NUM_CLASSES, dropout_rate=0.4).to(config.DEVICE)
                model.net.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
                model.net.eval()
        elif model_type == "ann":
            weights_path = os.path.join(checkpoints_dir, "ann_model.pth")
            model = LEGONeuralNetwork(input_dim=325, num_classes=config.NUM_CLASSES, hidden_dims=(256, 128))
            if os.path.exists(weights_path):
                model.net = MLPNet(input_dim=325, num_classes=config.NUM_CLASSES, hidden_dims=(256, 128)).to(config.DEVICE)
                model.net.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
                model.net.eval()
        elif model_type == "svm":
            p = os.path.join(checkpoints_dir, "svm_model.joblib")
            if os.path.exists(p):
                model = LEGOSVM.load(p)
        elif model_type == "perceptron":
            p = os.path.join(checkpoints_dir, "perceptron_model.joblib")
            if os.path.exists(p):
                model = LEGOPerceptron.load(p)
        elif model_type == "sgadaline":
            p = os.path.join(checkpoints_dir, "sgadaline_model.joblib")
            if os.path.exists(p):
                model = MulticlassSGAdaline.load(p)
        elif model_type == "decision_tree":
            p = os.path.join(checkpoints_dir, "decision_tree_model.joblib")
            if os.path.exists(p):
                model = LEGODecisionTree.load(p)

        if model is not None:
            self.models_cache[model_type] = model
        return model

    def predict_crop_with_model(self, crop_bgr, model_type=None, use_tta=False):
        """
        Classifies a single BGR crop using the specified model_type.
        Returns: (class_idx, confidence, class_name, probs_array)
        """
        m_type = model_type or self.model_type
        clf = self._get_or_load_model(m_type)

        if m_type == "cnn" and clf is not None and clf.net is not None:
            probs = clf.predict_proba_single_image(crop_bgr, use_tta=use_tta)
        elif clf is not None:
            feats_scaled = self.extractor.extract_and_scale_single(crop_bgr, scenario="shape_and_color")
            probs = clf.predict_proba(feats_scaled)[0]
            if use_tta and crop_bgr.shape[0] > crop_bgr.shape[1] * 1.15 and float(np.max(probs)) < 0.75:
                rot_crop = cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)
                feats_rot = self.extractor.extract_and_scale_single(rot_crop, scenario="shape_and_color")
                probs_rot = clf.predict_proba(feats_rot)[0]
                if float(np.max(probs_rot)) > float(np.max(probs)):
                    probs = probs_rot
        else:
            probs = np.ones(config.NUM_CLASSES, dtype=np.float32) / config.NUM_CLASSES

        c_idx = int(np.argmax(probs))
        conf = float(probs[c_idx])
        c_name = config.IDX_TO_CLASS[c_idx]
        return c_idx, conf, c_name, probs

    def predict_crop_all_models(self, crop_bgr, use_tta=False):
        """
        Evaluates all 6 trained models on the same piece crop for side-by-side comparison.
        """
        results = []
        # Extract scaled features once for the 5 feature-based models
        feats_scaled = self.extractor.extract_and_scale_single(crop_bgr, scenario="shape_and_color")
        feats_rot = None
        if use_tta and crop_bgr.shape[0] > crop_bgr.shape[1] * 1.15:
            rot_crop = cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)
            feats_rot = self.extractor.extract_and_scale_single(rot_crop, scenario="shape_and_color")

        for m_key in ["cnn", "svm", "ann", "perceptron", "sgadaline", "decision_tree"]:
            clf = self._get_or_load_model(m_key)
            if m_key == "cnn" and clf is not None and clf.net is not None:
                probs = clf.predict_proba_single_image(crop_bgr, use_tta=use_tta)
            elif clf is not None:
                probs = clf.predict_proba(feats_scaled)[0]
                if feats_rot is not None and float(np.max(probs)) < 0.75:
                    probs_rot = clf.predict_proba(feats_rot)[0]
                    if float(np.max(probs_rot)) > float(np.max(probs)):
                        probs = probs_rot
            else:
                continue

            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            top_class = config.IDX_TO_CLASS[top_idx]
            family = "Brick" if top_class.startswith("Brick") else "Plate"
            results.append({
                "model_key": m_key,
                "Modelo": MODEL_DISPLAY_NAMES[m_key],
                "Predicción": top_class,
                "Familia": family,
                "Confianza (%)": round(top_conf * 100.0, 2),
                "probs": probs
            })
        return results

    def process_image(self, image_input, segmentation_method="hsv_otsu", min_confidence=0.55,
                      filter_skin=False, only_main_piece=False, normalize_bg=False, use_tta=False,
                      save_output=True, show_plot=True, output_filename=None):
        """
        Executes end-to-end detection and classification on an input image.
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            img_name = os.path.splitext(os.path.basename(image_input))[0]
        else:
            img_bgr = image_input.copy()
            img_name = "sample_input"
            
        if img_bgr is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_input}")
            
        # 1. Segment image to detect LEGO pieces
        binary_mask, pieces = self.segmenter.detect_and_extract_pieces(
            img_bgr, method=segmentation_method, filter_skin=filter_skin, only_main_piece=only_main_piece
        )
        if len(pieces) == 0:
            binary_mask, pieces = self.segmenter.extract_whole_image_as_piece(img_bgr, filter_skin=filter_skin)
        
        detected_pieces = []
        labels = []
        confidences = []
        
        # 2. Classify each extracted piece
        for idx, piece in enumerate(pieces):
            crop_bgr = piece["crop"]
            if normalize_bg:
                crop_bgr = self.segmenter.normalize_crop_background(crop_bgr, piece.get("piece_mask"))
            
            class_idx, conf, raw_class_name, probs = self.predict_crop_with_model(
                crop_bgr, model_type=self.model_type, use_tta=use_tta
            )
                
            # Filter low confidence / Out-Of-Distribution pieces
            if conf < min_confidence:
                display_label = f"Desconocida ({raw_class_name}?)"
                is_confident = False
            else:
                display_label = raw_class_name
                is_confident = True
                
            labels.append(display_label)
            confidences.append(conf)
            
            detected_pieces.append({
                "piece_id": idx + 1,
                "class_name": display_label,
                "base_class": raw_class_name,
                "class_idx": class_idx,
                "confidence": float(conf),
                "probs": probs,
                "is_confident": is_confident,
                "tight_bbox": piece["tight_bbox"],
                "padded_bbox": piece["padded_bbox"],
                "contour_area": float(piece["area"]),
                "processed_crop": crop_bgr
            })
            
        # 3. Draw annotations
        annotated_overlay = self.segmenter.draw_detections(
            img_bgr, pieces, labels=labels, confidences=confidences
        )
        
        # 4. Create isolated segmented image (background masked out)
        masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=binary_mask)
        
        # 5. Composite 4-panel visual report
        if save_output or show_plot:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"1. Original ({img_name})", fontsize=11, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(binary_mask, cmap='gray')
            axes[1].set_title("2. Máscara Binaria (OpenCV HSV)", fontsize=11, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            axes[2].set_title("3. Pieza Segmentada Aislada", fontsize=11, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB))
            axes[3].set_title(f"4. Clasificación ({self.model_type.upper()})", fontsize=11, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            if save_output:
                out_dir = os.path.join(config.RESULTS_DIR, "predictions")
                os.makedirs(out_dir, exist_ok=True)
                if output_filename is None:
                    output_filename = f"detection_{img_name}_{segmentation_method}.png"
                save_path = os.path.join(out_dir, output_filename)
                plt.savefig(save_path, dpi=300)
                print(f"[Pipeline] Resultado guardado en: {save_path}")
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
        return {
            "image_name": img_name,
            "total_detected": len(detected_pieces),
            "pieces": detected_pieces,
            "mask": binary_mask,
            "overlay": annotated_overlay
        }
