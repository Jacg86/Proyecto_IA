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
import joblib
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

    def load_or_fit_scaler(self, scenario="shape_and_color"):
        """
        Loads cached StandardScaler for inference, or reconstructs it exactly from
        the precomputed training NPZ matrix and a small subset of training images.
        """
        scaler_file = os.path.join(
            config.DATA_DIR,
            "scaler_all.joblib" if scenario == "shape_and_color" else "scaler_shape.joblib"
        )
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            if scenario == "shape_and_color":
                self.scaler_all = scaler
            else:
                self.scaler_shape = scaler
            self.is_fitted = True
            return scaler

        npz_file = os.path.join(
            config.DATA_DIR,
            "features_shape_and_color.npz" if scenario == "shape_and_color" else "features_shape_only.npz"
        )
        splits_file = os.path.join(config.DATA_DIR, "dataset_splits.json")
        if os.path.exists(npz_file) and os.path.exists(splits_file) and os.path.isdir(config.CROPPED_IMAGES_DIR):
            z_train = np.load(npz_file)["X_train"]
            with open(splits_file, "r", encoding="utf-8") as f:
                splits = json.load(f)
            train_records = splits["train"]

            # Sample 60 evenly-spaced records across all classes to reconstruct exact affine parameters
            step = max(1, len(train_records) // 60)
            sample_indices = list(range(0, len(train_records), step))[:60]
            raw_list = []
            z_list = []
            for idx in sample_indices:
                rec = train_records[idx]
                fpath = os.path.join(config.CROPPED_IMAGES_DIR, rec["relative_path"])
                img = cv2.imread(fpath)
                if img is not None:
                    feats = self.extract_shape_features(img) if scenario == "shape_only" else self.extract_all_features(img)
                    raw_list.append(feats)
                    z_list.append(z_train[idx])

            if len(raw_list) >= 10:
                X_sub = np.array(raw_list, dtype=np.float64)
                Z_sub = np.array(z_list, dtype=np.float64)
                std_x = np.std(X_sub, axis=0)
                std_z = np.std(Z_sub, axis=0)
                scale = np.where(std_z > 1e-6, std_x / np.maximum(std_z, 1e-12), 1.0)
                mean = np.mean(X_sub, axis=0) - scale * np.mean(Z_sub, axis=0)

                scaler = StandardScaler()
                scaler.mean_ = mean.astype(np.float64)
                scaler.scale_ = scale.astype(np.float64)
                scaler.var_ = (scale ** 2).astype(np.float64)
                scaler.n_features_in_ = X_sub.shape[1]
                scaler.n_samples_seen_ = len(train_records)
                joblib.dump(scaler, scaler_file)
                if scenario == "shape_and_color":
                    self.scaler_all = scaler
                else:
                    self.scaler_shape = scaler
                self.is_fitted = True
                return scaler

        return None

    def extract_and_scale_single(self, img_bgr, scenario="shape_and_color"):
        """
        Extracts feature vector from a single BGR image crop and applies the training StandardScaler.
        """
        raw_feats = self.extract_shape_features(img_bgr) if scenario == "shape_only" else self.extract_all_features(img_bgr)
        raw_2d = raw_feats.reshape(1, -1)
        scaler = self.scaler_shape if (scenario == "shape_only" and self.is_fitted) else (
            self.scaler_all if self.is_fitted else self.load_or_fit_scaler(scenario=scenario)
        )
        if scaler is not None and hasattr(scaler, "mean_"):
            return scaler.transform(raw_2d).astype(np.float32)
        return raw_2d.astype(np.float32)

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
    joblib.dump(scaler_shape, os.path.join(config.DATA_DIR, "scaler_shape.joblib"))
    
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
    joblib.dump(scaler_all, os.path.join(config.DATA_DIR, "scaler_all.joblib"))
    
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

