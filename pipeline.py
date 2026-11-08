"""
End-to-End Pipeline module.
Integrates OpenCV HSV segmentation with the trained classifier.
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
from models.cnn_model import LEGOCNN
from models.svm_model import LEGOSVM

class LEGODetectionPipeline:
    def __init__(self, model_type="cnn"):
        self.model_type = model_type
        self.segmenter = HSVSegmenter(
            min_area=config.MIN_CONTOUR_AREA,
            margin=config.SEGMENTATION_MARGIN
        )
        self.classifier = None
        self._load_classifier()

    def _load_classifier(self):
        checkpoints_dir = os.path.join(config.MODELS_DIR, "checkpoints")
        
        if self.model_type == "cnn":
            weights_path = os.path.join(checkpoints_dir, "cnn_model.pth")
            self.classifier = LEGOCNN(num_classes=config.NUM_CLASSES)
            if os.path.exists(weights_path):
                from models.cnn_model import LEGOConvNet
                self.classifier.net = LEGOConvNet(num_classes=config.NUM_CLASSES, dropout_rate=0.4).to(config.DEVICE)
                self.classifier.net.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
                self.classifier.net.eval()
                print(f"[Pipeline] Clasificador CNN cargado desde: {weights_path}")
            else:
                print(f"[Pipeline Warning] Pesos CNN no encontrados en {weights_path}. Se inicializará o requerirá entrenamiento previo.")
        elif self.model_type == "svm":
            svm_path = os.path.join(checkpoints_dir, "svm_model.joblib")
            if os.path.exists(svm_path):
                self.classifier = LEGOSVM.load(svm_path)
                print(f"[Pipeline] Clasificador SVM cargado desde: {svm_path}")
            else:
                print(f"[Pipeline Warning] Modelo SVM no encontrado en {svm_path}.")

    def process_image(self, image_input, segmentation_method="hsv_otsu", min_confidence=0.55, filter_skin=False, save_output=True, output_filename=None):
        """
        Executes end-to-end detection and classification on an input image.
        
        Args:
            image_input: path to image (str) or loaded BGR image (np.ndarray)
            segmentation_method: 'hsv_otsu', 'hsv_adaptive', 'universal_adaptive', or 'bg_sub'
            min_confidence: minimum threshold to accept prediction; below this, flags as uncertain / unknown
            filter_skin: bool, whether to remove human skin tones (fingers/hands holding the brick)
            save_output: bool, whether to save visualization
            output_filename: str, optional output filename
            
        Returns:
            results_dict containing:
                'pieces': list of detected pieces with bounding box, predicted class, and confidence
                'mask': binary segmentation mask
                'overlay_image': image annotated with contours and classification labels
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
            img_bgr, method=segmentation_method, filter_skin=filter_skin
        )
        
        detected_pieces = []
        labels = []
        confidences = []
        
        # 2. Classify each extracted piece
        for idx, piece in enumerate(pieces):
            crop_bgr = piece["crop"]
            
            if self.model_type == "cnn" and self.classifier.net is not None:
                class_idx, conf, raw_class_name = self.classifier.predict_single_image(crop_bgr)
            elif self.model_type == "svm" and self.classifier is not None:
                from features.feature_extractor import LEGOFeatureExtractor
                extractor = LEGOFeatureExtractor()
                feats = extractor.extract_all_features(crop_bgr).reshape(1, -1)
                probs = self.classifier.predict_proba(feats)[0]
                class_idx = int(np.argmax(probs))
                conf = float(probs[class_idx])
                raw_class_name = config.IDX_TO_CLASS[class_idx]
            else:
                class_idx = 0
                conf = 0.50
                raw_class_name = config.IDX_TO_CLASS[class_idx]
                
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
                "is_confident": is_confident,
                "tight_bbox": piece["tight_bbox"],
                "padded_bbox": piece["padded_bbox"],
                "contour_area": float(piece["area"])
            })
            
        # 3. Draw annotations
        annotated_overlay = self.segmenter.draw_detections(
            img_bgr, pieces, labels=labels, confidences=confidences
        )
        
        # 4. Create isolated segmented image (background masked out)
        masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=binary_mask)
        
        # 5. Composite 4-panel visual report
        if save_output:
            out_dir = os.path.join(config.RESULTS_DIR, "predictions")
            os.makedirs(out_dir, exist_ok=True)
            if output_filename is None:
                output_filename = f"detection_{img_name}_{segmentation_method}.png"
                
            save_path = os.path.join(out_dir, output_filename)
            
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title("1. Fotografía Original", fontsize=11, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(binary_mask, cmap='gray')
            axes[1].set_title("2. Máscara Binaria (OpenCV HSV)", fontsize=11, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            axes[2].set_title("3. Pieza Segmentada Aislada", fontsize=11, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB))
            axes[3].set_title("4. Detección y Clasificación", fontsize=11, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[Pipeline] Resultado guardado en: {save_path}")
            
        return {
            "image_name": img_name,
            "total_detected": len(detected_pieces),
            "pieces": detected_pieces,
            "mask": binary_mask,
            "overlay": annotated_overlay
        }

