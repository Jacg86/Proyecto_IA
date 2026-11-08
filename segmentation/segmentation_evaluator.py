"""
Segmentation evaluation module.
Evaluates OpenCV segmentation methods against ground-truth reference masks,
calculates IoU, Dice, Pixel Accuracy, Precision, and Recall, and generates
visual comparison figures.
"""
import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from segmentation.hsv_segmenter import HSVSegmenter

def calculate_mask_metrics(y_true, y_pred):
    """
    Computes segmentation metrics between binary ground-truth (y_true)
    and binary prediction (y_pred), both containing values 0 or 255 (or 0 and 1).
    """
    true_bin = (y_true > 0).astype(np.uint8)
    pred_bin = (y_pred > 0).astype(np.uint8)
    
    intersection = np.logical_and(true_bin, pred_bin).sum()
    union = np.logical_or(true_bin, pred_bin).sum()
    
    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2.0 * intersection / (true_bin.sum() + pred_bin.sum())) if (true_bin.sum() + pred_bin.sum()) > 0 else 1.0
    
    tp = intersection
    fp = np.logical_and(pred_bin == 1, true_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, true_bin == 1).sum()
    tn = np.logical_and(pred_bin == 0, true_bin == 0).sum()
    total = true_bin.size
    
    pixel_acc = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    return {
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": pixel_acc,
        "precision": precision,
        "recall": recall
    }

def get_ground_truth_mask_and_bbox(base_img, crop_img):
    """
    Derives ground truth bounding box and pixel-level mask from the exact matching crop.
    """
    res = cv2.matchTemplate(base_img, crop_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.90:
        return None, None
        
    h_c, w_c = crop_img.shape[:2]
    x_c, y_c = max_loc
    
    # Generate ground-truth mask inside the crop window
    gt_mask = np.zeros(base_img.shape[:2], dtype=np.uint8)
    
    # Segment piece inside the crop (high contrast local region)
    crop_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    _, s_mask = cv2.threshold(crop_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v_mask = cv2.threshold(crop_hsv[:, :, 2], 130, 255, cv2.THRESH_BINARY_INV)
    local_mask = cv2.bitwise_or(s_mask, v_mask)
    
    # Morphological cleanup on local mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, k)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, k, iterations=2)
    
    # Keep only the largest contour in the crop window (the piece itself)
    cnts, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    piece_only_mask = np.zeros_like(local_mask)
    if cnts:
        best_cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(piece_only_mask, [best_cnt], -1, 255, -1)
    else:
        piece_only_mask = local_mask
        
    gt_mask[y_c:y_c+h_c, x_c:x_c+w_c] = piece_only_mask
    gt_bbox = (x_c, y_c, w_c, h_c)
    return gt_mask, gt_bbox

def evaluate_segmentation(num_samples=40):
    print("=" * 60)
    print("ETAPA 3: EVALUACIÓN CUANTITATIVA DE SEGMENTACIÓN OPENCV")
    print("=" * 60)
    
    segmenter = HSVSegmenter()
    
    # Curate sample pairs from Base Images and Cropped Images across diverse classes
    base_classes = [d for d in os.listdir(config.BASE_IMAGES_DIR) if os.path.isdir(os.path.join(config.BASE_IMAGES_DIR, d))]
    
    eval_pairs = []
    for cls_name in sorted(base_classes):
        cls_base = os.path.join(config.BASE_IMAGES_DIR, cls_name)
        cls_crop = os.path.join(config.CROPPED_IMAGES_DIR, cls_name)
        files = [f for f in os.listdir(cls_base) if f.lower().endswith('.jpg')]
        # Select 2-3 samples per class
        for f in files[:3]:
            crop_f = "1_" + f[2:]
            crop_path = os.path.join(cls_crop, crop_f)
            if os.path.exists(crop_path):
                eval_pairs.append((cls_name, os.path.join(cls_base, f), crop_path))
                if len(eval_pairs) >= num_samples:
                    break
        if len(eval_pairs) >= num_samples:
            break
            
    print(f"Total imágenes de referencia para validación: {len(eval_pairs)}")
    
    metrics_by_method = {
        "hsv_otsu": {"iou": [], "dice": [], "pixel_accuracy": [], "precision": [], "recall": []},
        "hsv_adaptive": {"iou": [], "dice": [], "pixel_accuracy": [], "precision": [], "recall": []},
        "bg_subtraction": {"iou": [], "dice": [], "pixel_accuracy": [], "precision": [], "recall": []}
    }
    
    visual_examples = []
    
    for idx, (cls_name, base_p, crop_p) in enumerate(eval_pairs):
        base_img = cv2.imread(base_p)
        crop_img = cv2.imread(crop_p)
        
        gt_mask, gt_bbox = get_ground_truth_mask_and_bbox(base_img, crop_img)
        if gt_mask is None:
            continue
            
        # 1. HSV Otsu
        mask_otsu, pieces_otsu = segmenter.detect_and_extract_pieces(base_img, method='hsv_otsu')
        m_otsu = calculate_mask_metrics(gt_mask, mask_otsu)
        for k, v in m_otsu.items():
            metrics_by_method["hsv_otsu"][k].append(v)
            
        # 2. HSV Adaptive
        mask_adapt, pieces_adapt = segmenter.detect_and_extract_pieces(base_img, method='hsv_adaptive')
        m_adapt = calculate_mask_metrics(gt_mask, mask_adapt)
        for k, v in m_adapt.items():
            metrics_by_method["hsv_adaptive"][k].append(v)
            
        # 3. Background Subtraction
        mask_bg, pieces_bg = segmenter.detect_and_extract_pieces(base_img, method='bg_sub')
        m_bg = calculate_mask_metrics(gt_mask, mask_bg)
        for k, v in m_bg.items():
            metrics_by_method["bg_subtraction"][k].append(v)
            
        # Save a few diverse visual comparisons
        if idx < 6:
            overlay_otsu = segmenter.draw_detections(base_img, pieces_otsu, labels=[cls_name])
            visual_examples.append({
                "class": cls_name,
                "base_img": base_img,
                "gt_mask": gt_mask,
                "pred_mask": mask_otsu,
                "overlay": overlay_otsu,
                "iou": m_otsu["iou"],
                "dice": m_otsu["dice"]
            })
            
    # Compute aggregates
    summary = {}
    for method, metrics in metrics_by_method.items():
        summary[method] = {
            "mean_iou": float(np.mean(metrics["iou"])),
            "std_iou": float(np.std(metrics["iou"])),
            "mean_dice": float(np.mean(metrics["dice"])),
            "mean_pixel_accuracy": float(np.mean(metrics["pixel_accuracy"])),
            "mean_precision": float(np.mean(metrics["precision"])),
            "mean_recall": float(np.mean(metrics["recall"]))
        }
        
    print("\n" + "=" * 60)
    print("RESULTADOS CUANTITATIVOS DE SEGMENTACIÓN:")
    print("=" * 60)
    print(f"{'Método':20s} | {'IoU':8s} | {'Dice':8s} | {'Pixel Acc':10s} | {'Precision':10s} | {'Recall':8s}")
    print("-" * 75)
    for method, s in summary.items():
        print(f"{method:20s} | {s['mean_iou']:8.4f} | {s['mean_dice']:8.4f} | {s['mean_pixel_accuracy']:10.4f} | {s['mean_precision']:10.4f} | {s['mean_recall']:8.4f}")
        
    # Save visual comparison plots
    out_dir = os.path.join(config.RESULTS_DIR, "segmentation")
    os.makedirs(out_dir, exist_ok=True)
    
    for i, ex in enumerate(visual_examples):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Convert BGR to RGB for matplotlib
        axes[0].imshow(cv2.cvtColor(ex["base_img"], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original: {ex['class']}", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(ex["gt_mask"], cmap='gray')
        axes[1].set_title("Máscara Ground Truth", fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(ex["pred_mask"], cmap='gray')
        axes[2].set_title(f"Máscara HSV Otsu\n(IoU={ex['iou']:.3f}, Dice={ex['dice']:.3f})", fontsize=10, fontweight='bold')
        axes[2].axis('off')
        
        axes[3].imshow(cv2.cvtColor(ex["overlay"], cv2.COLOR_BGR2RGB))
        axes[3].set_title("Detección y Contornos", fontsize=11, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"segmentation_sample_{i+1}_{ex['class']}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    # Save summary JSON
    metrics_path = os.path.join(out_dir, "segmentation_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"\nResultados y figuras guardados en: {out_dir}")
    
    return summary

if __name__ == "__main__":
    evaluate_segmentation(num_samples=35)

