"""
Automated unit and integration test suite for the LEGO project.
Tests dataset integrity, preprocessing, segmentation, feature extraction,
model loading, and end-to-end pipeline inference.
"""
import os
import sys
import pytest
import numpy as np
import cv2
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import create_or_load_splits, preprocess_image_crop
from segmentation.hsv_segmenter import HSVSegmenter
from features.feature_extractor import LEGOFeatureExtractor
from pipeline import LEGODetectionPipeline

def test_dataset_and_classes():
    assert len(config.ALL_CLASSES) == 20
    assert len(config.BRICK_CLASSES) == 9
    assert len(config.PLATE_CLASSES) == 11
    assert os.path.exists(config.DATASET_ROOT)

def test_data_splits_leak_free():
    splits = create_or_load_splits()
    train_recs = splits["train"]
    val_recs = splits["val"]
    test_recs = splits["test"]
    
    assert len(train_recs) > 0
    assert len(val_recs) > 0
    assert len(test_recs) > 0
    assert len(train_recs) + len(val_recs) + len(test_recs) == 4580
    
    # Check no overlap in filenames
    train_files = set(r["relative_path"] for r in train_recs)
    val_files = set(r["relative_path"] for r in val_recs)
    test_files = set(r["relative_path"] for r in test_recs)
    
    assert len(train_files.intersection(val_files)) == 0
    assert len(train_files.intersection(test_files)) == 0
    assert len(val_files.intersection(test_files)) == 0

def test_hsv_segmenter():
    segmenter = HSVSegmenter()
    # Test on dummy image
    dummy_img = np.full((480, 640, 3), (240, 190, 160), dtype=np.uint8)
    # Draw blue piece in center
    cv2.rectangle(dummy_img, (200, 150), (350, 280), (220, 50, 20), -1)
    
    mask, pieces = segmenter.detect_and_extract_pieces(dummy_img, method='hsv_otsu')
    assert mask is not None
    assert mask.shape == (480, 640)
    assert len(pieces) >= 1
    assert pieces[0]["area"] > 500

def test_feature_extractor():
    extractor = LEGOFeatureExtractor()
    dummy_crop = np.full((64, 64, 3), (200, 100, 50), dtype=np.uint8)
    
    shape_feats = extractor.extract_shape_features(dummy_crop)
    all_feats = extractor.extract_all_features(dummy_crop)
    
    assert len(shape_feats) == 301
    assert len(all_feats) == 325

def test_trained_model_checkpoints():
    checkpoints_dir = os.path.join(config.MODELS_DIR, "checkpoints")
    
    # Check CNN
    cnn_path = os.path.join(checkpoints_dir, "cnn_model.pth")
    assert os.path.exists(cnn_path)
    from models.cnn_model import LEGOConvNet
    net = LEGOConvNet(num_classes=20)
    state_dict = torch.load(cnn_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    
    # Check SVM
    svm_path = os.path.join(checkpoints_dir, "svm_model.joblib")
    assert os.path.exists(svm_path)
    svm = joblib.load(svm_path)
    assert hasattr(svm, "predict")

def test_pipeline_end_to_end():
    pipeline = LEGODetectionPipeline(model_type="cnn")
    
    # Run on one sample from Base Images
    sample_dir = os.path.join(config.BASE_IMAGES_DIR, "Brick_2x2_L")
    files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
    assert len(files) > 0
    sample_path = os.path.join(sample_dir, files[0])
    
    result = pipeline.process_image(sample_path, segmentation_method="hsv_otsu", save_output=False)
    assert result["total_detected"] >= 1
    assert "pieces" in result
    assert "mask" in result
    assert "overlay" in result
    assert result["pieces"][0]["confidence"] > 0.0

