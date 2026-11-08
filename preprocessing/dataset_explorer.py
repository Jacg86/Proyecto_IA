"""
Dataset Explorer module.
Performs thorough exploratory data analysis on the LEGO dataset, checks integrity,
duplicates, distributions, and saves visual summaries.
"""
import os
import sys
import re
import json
import hashlib
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def compute_file_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def extract_timestamp_id(filename):
    """
    Extracts the unique timestamp identifier from LEGO image filenames.
    e.g. '1_Brick_2x2_L_180708213430.jpg' or '2_Brick_2x2_L_180708213430.jpg' -> '180708213430'
    """
    match = re.search(r'(\d{10,14})\.jpg', filename)
    if match:
        return match.group(1)
    return filename

def explore_dataset():
    print("=" * 60)
    print("ETAPA 1: EXPLORACIÓN Y ANÁLISIS DEL DATASET LEGO")
    print("=" * 60)
    
    results = {
        "base_images": {},
        "cropped_images": {},
        "correspondence": {},
        "duplicates": {"base": [], "cropped": []},
        "corrupt": {"base": [], "cropped": []}
    }
    
    # 1. Analyze Cropped Images (Primary classification source)
    print("\n--- Analizando Cropped Images (20 categorías) ---")
    crop_counts = {}
    crop_resolutions = []
    crop_hashes = {}
    crop_records = []
    
    for cls_name in config.ALL_CLASSES:
        cls_dir = os.path.join(config.CROPPED_IMAGES_DIR, cls_name)
        if not os.path.exists(cls_dir):
            print(f"[ALERTA] No existe la carpeta {cls_dir}")
            crop_counts[cls_name] = 0
            continue
        
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        crop_counts[cls_name] = len(files)
        
        for f in files:
            fpath = os.path.join(cls_dir, f)
            h = compute_file_md5(fpath)
            if h in crop_hashes:
                results["duplicates"]["cropped"].append((fpath, crop_hashes[h]))
            else:
                crop_hashes[h] = fpath
                
            img = cv2.imread(fpath)
            if img is None:
                results["corrupt"]["cropped"].append(fpath)
                continue
            
            h_img, w_img = img.shape[:2]
            crop_resolutions.append((w_img, h_img))
            ts_id = extract_timestamp_id(f)
            crop_records.append({
                "class": cls_name,
                "file": f,
                "path": fpath,
                "width": w_img,
                "height": h_img,
                "timestamp_id": ts_id
            })
            
    print(f"Total imágenes recortadas válidas: {len(crop_records)}")
    print(f"Duplicados exactos detectados: {len(results['duplicates']['cropped'])}")
    print(f"Archivos corruptos detectados: {len(results['corrupt']['cropped'])}")
    
    # 2. Analyze Base Images (Segmentation source)
    print("\n--- Analizando Base Images ---")
    base_counts = {}
    base_records = []
    base_hashes = {}
    
    for item in os.listdir(config.BASE_IMAGES_DIR):
        cls_dir = os.path.join(config.BASE_IMAGES_DIR, item)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        base_counts[item] = len(files)
        for f in files:
            fpath = os.path.join(cls_dir, f)
            h = compute_file_md5(fpath)
            if h in base_hashes:
                results["duplicates"]["base"].append((fpath, base_hashes[h]))
            else:
                base_hashes[h] = fpath
                
            img = cv2.imread(fpath)
            if img is None:
                results["corrupt"]["base"].append(fpath)
                continue
            
            h_img, w_img = img.shape[:2]
            ts_id = extract_timestamp_id(f)
            base_records.append({
                "class": item,
                "file": f,
                "path": fpath,
                "width": w_img,
                "height": h_img,
                "timestamp_id": ts_id
            })
            
    print(f"Total imágenes base válidas: {len(base_records)}")
    print(f"Categorías presentes en Base Images: {len(base_counts)}")
    print(f"Duplicados en Base Images: {len(results['duplicates']['base'])}")
    print(f"Archivos corruptos en Base Images: {len(results['corrupt']['base'])}")
    
    # 3. Analyze relationship and correspondence
    base_ts = {r["timestamp_id"]: r for r in base_records}
    crop_ts = {r["timestamp_id"]: r for r in crop_records}
    
    common_ts = set(base_ts.keys()).intersection(set(crop_ts.keys()))
    print(f"\nImágenes con correspondencia directa Base <-> Cropped: {len(common_ts)}")
    print(f"Tasa de correspondencia sobre Base Images: {len(common_ts) / max(1, len(base_records)) * 100:.2f}%")
    
    # 4. Generate Visualizations
    out_dir = os.path.join(config.RESULTS_DIR, "exploration")
    os.makedirs(out_dir, exist_ok=True)
    
    # Class distribution plot
    fig, ax = plt.subplots(figsize=(14, 6))
    classes = list(crop_counts.keys())
    counts = [crop_counts[c] for c in classes]
    colors = ['#1f77b4' if c.startswith('Brick') else '#ff7f0e' for c in classes]
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title("Distribución de Clases en Cropped Images (Total: 4,580 piezas)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Categoría LEGO", fontsize=12)
    ax.set_ylabel("Cantidad de Imágenes", fontsize=12)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, int(yval), ha='center', va='bottom', fontsize=8)
        
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Familia Brick (9 clases)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Familia Plate (11 clases)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    dist_plot_path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Gráfico de distribución guardado en: {dist_plot_path}")
    
    # Resolution scatter & aspect ratio histogram
    widths = [w for w, h in crop_resolutions]
    heights = [h for w, h in crop_resolutions]
    aspect_ratios = [w / float(h) for w, h in crop_resolutions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(widths, heights, alpha=0.3, color='purple', edgecolors='none', s=20)
    ax1.set_title("Resoluciones de Cropped Images (Ancho vs Alto)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Ancho (píxeles)", fontsize=11)
    ax1.set_ylabel("Alto (píxeles)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.hist(aspect_ratios, bins=30, color='teal', edgecolor='black', alpha=0.8)
    ax2.set_title("Distribución de Relación de Aspecto (W / H)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Aspect Ratio", fontsize=11)
    ax2.set_ylabel("Frecuencia", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    res_plot_path = os.path.join(out_dir, "resolution_and_aspect_ratio.png")
    plt.savefig(res_plot_path, dpi=300)
    plt.close()
    print(f"Gráfico de resoluciones guardado en: {res_plot_path}")
    
    # Save JSON summary
    summary = {
        "total_cropped_images": len(crop_records),
        "total_base_images": len(base_records),
        "classes_count": len(config.ALL_CLASSES),
        "cropped_class_counts": crop_counts,
        "base_class_counts": base_counts,
        "matching_timestamps_count": len(common_ts),
        "exact_duplicates_count": len(results["duplicates"]["cropped"]),
        "corrupt_files_count": len(results["corrupt"]["cropped"]),
        "resolutions_summary": {
            "min_w": int(min(widths)),
            "max_w": int(max(widths)),
            "mean_w": float(np.mean(widths)),
            "min_h": int(min(heights)),
            "max_h": int(max(heights)),
            "mean_h": float(np.mean(heights)),
        }
    }
    
    summary_path = os.path.join(out_dir, "dataset_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"Resumen JSON guardado en: {summary_path}")
    
    return summary, crop_records, base_records

if __name__ == "__main__":
    explore_dataset()
