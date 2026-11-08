"""
Feature extraction module for LEGO piece classification.
Extracts geometric, shape (Hu moments, compactness, solidity, aspect ratio),
HOG (Histogram of Oriented Gradients), and color (HSV histograms) descriptors.
Supports two experimental scenarios:
1. Shape & Geometry only
2. Shape, Geometry & Color
"""
import os
import sys
import json
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import create_or_load_splits, preprocess_image_crop

class LEGOFeatureExtractor:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.scaler_shape = StandardScaler()
        self.scaler_all = StandardScaler()
        self.is_fitted = False

    def extract_shape_features(self, img_bgr):
        """
        Extracts pure shape and geometric descriptors:
        - Contour area, perimeter, circularity / compactness
        - Bounding box aspect ratio, extent, solidity
        - 7 Hu Moments (scale, translation, and rotation invariant)
        - HOG features (captures edges, shapes, and stud patterns)
        """
        # Resize to standard dimensions
        resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Binarize to extract piece contour inside the crop
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # If binary mask has > 70% pixels, invert it
        if np.count_nonzero(binary) > 0.70 * binary.size:
            binary = cv2.bitwise_not(binary)
            
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            perimeter = float(cv2.arcLength(c, True))
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / max(1.0, float(h))
            extent = area / max(1.0, float(w * h))
            
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(1.0, hull_area)
            circularity = (4.0 * np.pi * area) / max(1.0, (perimeter * perimeter))
            
            # Hu Moments
            moments = cv2.moments(c)
            hu = cv2.HuMoments(moments).flatten()
            hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        else:
            area = 0.0
            perimeter = 0.0
            aspect_ratio = 1.0
            extent = 0.0
            solidity = 0.0
            circularity = 0.0
            hu_log = np.zeros(7, dtype=np.float32)
            
        # Normalize area and perimeter by image dimensions
        norm_area = area / float(self.target_size[0] * self.target_size[1])
        norm_perimeter = perimeter / float(2 * (self.target_size[0] + self.target_size[1]))
        
        geom_features = np.array([
            norm_area, norm_perimeter, aspect_ratio, extent, solidity, circularity
        ], dtype=np.float32)
        
        # HOG descriptor using skimage
        hog_feat = hog(
            gray,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        ).astype(np.float32)
        
        return np.concatenate([geom_features, hu_log, hog_feat])

    def extract_color_features(self, img_bgr, bins=(8, 8, 8)):
        """
        Extracts normalized HSV color histogram features:
        - 8 bins for Hue (0-180)
        - 8 bins for Saturation (0-256)
        - 8 bins for Value (0-256)
        Total color features: 24 (or concatenated 1D histograms)
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        h_hist = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256]).flatten()
        
        # Normalize histograms (L1 norm so sum=1.0)
        h_hist /= max(1e-7, h_hist.sum())
        s_hist /= max(1e-7, s_hist.sum())
        v_hist /= max(1e-7, v_hist.sum())
        
        return np.concatenate([h_hist, s_hist, v_hist])

    def extract_all_features(self, img_bgr):
        """
        Extracts combined shape, geometric, and color features.
        """
        shape_feat = self.extract_shape_features(img_bgr)
        color_feat = self.extract_color_features(img_bgr)
        return np.concatenate([shape_feat, color_feat])

    def extract_dataset_features(self, split_records, scenario="shape_and_color"):
        """
        Extracts features for an entire split list of records.
        """
        X = []
        y = []
        for rec in split_records:
            fpath = os.path.join(config.CROPPED_IMAGES_DIR, rec["relative_path"])
            img = cv2.imread(fpath)
            if img is None:
                continue
            if scenario == "shape_only":
                feats = self.extract_shape_features(img)
            else:
                feats = self.extract_all_features(img)
            X.append(feats)
            y.append(rec["class_idx"])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def prepare_and_cache_features(force_recompute=False):
    """
    Extracts and standardizes features for both scenarios (shape_only and shape_and_color)
    across Train, Val, and Test splits. Saves to data/ for rapid reuse.
    """
    out_file_shape = os.path.join(config.DATA_DIR, "features_shape_only.npz")
    out_file_all = os.path.join(config.DATA_DIR, "features_shape_and_color.npz")
    
    if os.path.exists(out_file_shape) and os.path.exists(out_file_all) and not force_recompute:
        print("Cargando características precalculadas desde caché...")
        data_shape = np.load(out_file_shape)
        data_all = np.load(out_file_all)
        return data_shape, data_all
        
    print("\n" + "=" * 60)
    print("ETAPA 4: EXTRACCIÓN DE CARACTERÍSTICAS (Forma vs Forma+Color)")
    print("=" * 60)
    
    splits = create_or_load_splits()
    extractor = LEGOFeatureExtractor()
    
    # --- Scenario 1: Shape Only ---
    print("\nExtrayendo características Escenario 1: Solo Forma y Geometría (Hu, HOG, métricas)...")
    X_train_shape, y_train = extractor.extract_dataset_features(splits["train"], scenario="shape_only")
    X_val_shape, y_val = extractor.extract_dataset_features(splits["val"], scenario="shape_only")
    X_test_shape, y_test = extractor.extract_dataset_features(splits["test"], scenario="shape_only")
    
    print(f"Dimensión de vector de forma: {X_train_shape.shape[1]} características")
    # Fit scaler on train only!
    scaler_shape = StandardScaler()
    X_train_shape_s = scaler_shape.fit_transform(X_train_shape)
    X_val_shape_s = scaler_shape.transform(X_val_shape)
    X_test_shape_s = scaler_shape.transform(X_test_shape)
    
    np.savez_compressed(
        out_file_shape,
        X_train=X_train_shape_s, y_train=y_train,
        X_val=X_val_shape_s, y_val=y_val,
        X_test=X_test_shape_s, y_test=y_test
    )
    print(f"Características de forma guardadas en: {out_file_shape}")
    
    # --- Scenario 2: Shape + Color ---
    print("\nExtrayendo características Escenario 2: Forma + Geometría + Color (HSV)...")
    X_train_all, _ = extractor.extract_dataset_features(splits["train"], scenario="shape_and_color")
    X_val_all, _ = extractor.extract_dataset_features(splits["val"], scenario="shape_and_color")
    X_test_all, _ = extractor.extract_dataset_features(splits["test"], scenario="shape_and_color")
    
    print(f"Dimensión de vector combinado (forma+color): {X_train_all.shape[1]} características")
    scaler_all = StandardScaler()
    X_train_all_s = scaler_all.fit_transform(X_train_all)
    X_val_all_s = scaler_all.transform(X_val_all)
    X_test_all_s = scaler_all.transform(X_test_all)
    
    np.savez_compressed(
        out_file_all,
        X_train=X_train_all_s, y_train=y_train,
        X_val=X_val_all_s, y_val=y_val,
        X_test=X_test_all_s, y_test=y_test
    )
    print(f"Características combinadas guardadas en: {out_file_all}")
    
    return np.load(out_file_shape), np.load(out_file_all)

if __name__ == "__main__":
    prepare_and_cache_features(force_recompute=True)

