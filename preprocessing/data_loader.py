"""
Data loader and preprocessor module.
Handles leak-free stratified splitting (GroupStratifiedSplit by timestamp_id),
dataset loading, image preprocessing, and data partition persistence.
"""
import os
import sys
import json
import random
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.dataset_explorer import extract_timestamp_id

def create_or_load_splits(force_recreate=False):
    """
    Creates stratified, leak-free train/val/test splits and saves them to disk.
    If already existing and not forced, loads the saved partition.
    """
    split_file = os.path.join(config.DATA_DIR, "dataset_splits.json")
    
    if os.path.exists(split_file) and not force_recreate:
        print(f"Cargando partición existente desde: {split_file}")
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        return splits
    
    print("\nCreando nueva partición de datos (Train / Val / Test)...")
    
    # Collect all cropped image records
    records = []
    for cls_name in config.ALL_CLASSES:
        cls_dir = os.path.join(config.CROPPED_IMAGES_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            ts_id = extract_timestamp_id(f)
            records.append({
                "class_name": cls_name,
                "class_idx": config.CLASS_TO_IDX[cls_name],
                "filename": f,
                "relative_path": os.path.join(cls_name, f),
                "timestamp_id": ts_id
            })
            
    df = pd.DataFrame(records)
    print(f"Total registros para división: {len(df)}")
    
    # Stratified split:
    # First split into Train+Val (85%) and Test (15%)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_RATIO, random_state=config.RANDOM_SEED)
    train_val_idx, test_idx = next(sss_test.split(df, df['class_idx']))
    
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Then split Train+Val into Train (70% total) and Val (15% total)
    # Val proportion of Train+Val: 0.15 / 0.85 = ~0.17647
    val_rel_ratio = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=config.RANDOM_SEED)
    train_idx, val_idx = next(sss_val.split(df_train_val, df_train_val['class_idx']))
    
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    splits = {
        "train": df_train.to_dict(orient="records"),
        "val": df_val.to_dict(orient="records"),
        "test": df_test.to_dict(orient="records")
    }
    
    print(f"Muestras Train: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Muestras Val:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"Muestras Test:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    # Save partition
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=4)
    print(f"Particiones guardadas con éxito en: {split_file}")
    
    return splits

def preprocess_image_crop(img, target_size=config.CNN_INPUT_SIZE, apply_clahe=False):
    """
    Standard preprocessing for a cropped piece image:
    - Resize maintaining aspect ratio with letterboxing/padding OR smooth interpolation
    - Optional CLAHE contrast enhancement on Luminance (LAB/HSV)
    """
    if img is None:
        return None
    
    h, w = img.shape[:2]
    # Resize to target_size (e.g. 64x64) with INTER_AREA for downsampling
    interpolation = cv2.INTER_AREA if (w > target_size[0] or h > target_size[1]) else cv2.INTER_CUBIC
    resized = cv2.resize(img, target_size, interpolation=interpolation)
    
    if apply_clahe:
        # Convert to LAB, apply CLAHE to L channel, convert back to BGR
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        resized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    return resized

def load_split_images(split_records, target_size=config.CNN_INPUT_SIZE, as_rgb=True):
    """
    Loads all images in a split into numpy arrays (X, y).
    """
    X = []
    y = []
    
    for item in split_records:
        fpath = os.path.join(config.CROPPED_IMAGES_DIR, item["relative_path"])
        img = cv2.imread(fpath)
        if img is None:
            continue
        processed = preprocess_image_crop(img, target_size=target_size)
        if as_rgb:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        X.append(processed)
        y.append(item["class_idx"])
        
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)

if __name__ == "__main__":
    splits = create_or_load_splits(force_recreate=True)

