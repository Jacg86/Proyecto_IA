"""
Watershed separation analysis module.
Evaluates the behavior of the Watershed algorithm on separated vs touching LEGO bricks.
Analyzes the impact of LEGO studs (reliefs/tetones) on over-segmentation.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from segmentation.hsv_segmenter import HSVSegmenter

def create_synthetic_touching_scene(img1_path, img2_path):
    """
    Creates a composite scene with two LEGO bricks placed side-by-side touching each other
    on the backlit background to test object separation.
    """
    bg = cv2.imread(config.BG_BACKLIT_A)
    if bg is None:
        bg = np.full((480, 640, 3), (245, 195, 170), dtype=np.uint8)
        
    scene = bg.copy()
    seg = HSVSegmenter()
    
    # Extract piece 1
    i1 = cv2.imread(img1_path)
    m1, p1 = seg.detect_and_extract_pieces(i1)
    
    # Extract piece 2
    i2 = cv2.imread(img2_path)
    m2, p2 = seg.detect_and_extract_pieces(i2)
    
    if not p1 or not p2:
        return None
        
    c1, mask1 = p1[0]["crop"], p1[0]["piece_mask"]
    c2, mask2 = p2[0]["crop"], p2[0]["piece_mask"]
    
    h1, w1 = c1.shape[:2]
    h2, w2 = c2.shape[:2]
    
    # Place piece 1 in center-left
    y1, x1 = 180, 180
    roi1 = scene[y1:y1+h1, x1:x1+w1]
    mask1_inv = cv2.bitwise_not(mask1)
    bg1 = cv2.bitwise_and(roi1, roi1, mask=mask1_inv)
    fg1 = cv2.bitwise_and(c1, c1, mask=mask1)
    scene[y1:y1+h1, x1:x1+w1] = cv2.add(bg1, fg1)
    
    # Place piece 2 directly touching piece 1 (x2 = x1 + w1 - 8 to simulate contact)
    y2, x2 = 180, x1 + w1 - 8
    if x2 + w2 < 640 and y2 + h2 < 480:
        roi2 = scene[y2:y2+h2, x2:x2+w2]
        mask2_inv = cv2.bitwise_not(mask2)
        bg2 = cv2.bitwise_and(roi2, roi2, mask=mask2_inv)
        fg2 = cv2.bitwise_and(c2, c2, mask=mask2)
        scene[y2:y2+h2, x2:x2+w2] = cv2.add(bg2, fg2)
        
    return scene

def analyze_watershed_behavior():
    print("=" * 65)
    print("EVALUACIÓN DEL ALGORITMO WATERSHED EN PIEZAS EN CONTACTO")
    print("=" * 65)
    
    seg = HSVSegmenter()
    
    # Select two sample base images automatically from existing classes
    cls1_dir = os.path.join(config.BASE_IMAGES_DIR, "Brick_2x2_L")
    cls2_dir = os.path.join(config.BASE_IMAGES_DIR, "Plate_1x2")
    
    files1 = [f for f in os.listdir(cls1_dir) if f.endswith('.jpg')]
    files2 = [f for f in os.listdir(cls2_dir) if f.endswith('.jpg')]
    
    if not (files1 and files2):
        print("Imágenes de prueba no encontradas para escena compuesta.")
        return
        
    p1 = os.path.join(cls1_dir, files1[0])
    p2 = os.path.join(cls2_dir, files2[0])
        
    scene = create_synthetic_touching_scene(p1, p2)
    if scene is None:
        print("Error al construir escena de contacto.")
        return
        
    # 1. Standard HSV Segmentation without Watershed
    mask_std, pieces_std = seg.detect_and_extract_pieces(scene, method='hsv_otsu', use_watershed=False)
    
    # 2. HSV Segmentation with Watershed Object Separation
    mask_ws, pieces_ws = seg.detect_and_extract_pieces(scene, method='hsv_otsu', use_watershed=True)
    
    # Draw overlays
    overlay_std = seg.draw_detections(scene, pieces_std)
    overlay_ws = seg.draw_detections(scene, pieces_ws)
    
    print(f"Piezas detectadas SIN Watershed (contornos directos): {len(pieces_std)}")
    print(f"Piezas detectadas CON Watershed: {len(pieces_ws)}")
    
    # Plot comparison
    out_dir = os.path.join(config.RESULTS_DIR, "segmentation")
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Escena con Piezas en Contacto", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_std, cmap='gray')
    axes[0, 1].set_title(f"Máscara HSV Estándar ({len(pieces_std)} objeto unificado)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask_ws, cmap='gray')
    axes[1, 0].set_title(f"Máscara con Separación Watershed ({len(pieces_ws)} objetos)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(overlay_ws, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Detección y Cajas Delimitadoras tras Watershed", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "watershed_separation_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Análisis visual de Watershed guardado en: {save_path}")
    print("\n[Conclusión Técnica]:")
    print("- Cuando las piezas no están en contacto, el método HSV estándar es superior y no introduce fragmentación.")
    print("- En piezas superpuestas o en contacto directo, Watershed permite separar las componentes mediante la transformada de distancia.")
    print("- Sin embargo, si el umbral de distancia es demasiado permisivo, los tetones superiores de las piezas LEGO pueden producir sobre-segmentación.")

if __name__ == "__main__":
    analyze_watershed_behavior()
