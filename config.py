"""
Configuration module for the LEGO piece segmentation and classification project.
Defines paths, class names, constants, and hyperparameters.
"""
import os
import torch

# Base project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "LEGO_BRICKS")
BASE_IMAGES_DIR = os.path.join(DATASET_ROOT, "Base Images")
CROPPED_IMAGES_DIR = os.path.join(DATASET_ROOT, "Cropped Images")
BG_BACKLIT_A = os.path.join(DATASET_ROOT, "background_backlit_A.jpg")
BG_BACKLIT_B = os.path.join(DATASET_ROOT, "background_backlit_B.jpg")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MASKS_DIR = os.path.join(PROJECT_ROOT, "masks")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 20 LEGO Classes organized by families
BRICK_CLASSES = [
    "Brick_1x1", "Brick_1x2", "Brick_1x3", "Brick_1x4",
    "Brick_2x2", "Brick_2x2_L", "Brick_2x2_Slope", "Brick_2x3", "Brick_2x4"
]

PLATE_CLASSES = [
    "Plate_1x1", "Plate_1x1_Round", "Plate_1x1_Slope",
    "Plate_1x2", "Plate_1x2_Grill", "Plate_1x3", "Plate_1x4",
    "Plate_2x2", "Plate_2x2_L", "Plate_2x3", "Plate_2x4"
]

ALL_CLASSES = sorted(BRICK_CLASSES + PLATE_CLASSES)
NUM_CLASSES = len(ALL_CLASSES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(ALL_CLASSES)}

# Reproducibility seed
RANDOM_SEED = 42

# Train / Validation / Test split proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image dimensions
BASE_IMAGE_SIZE = (640, 480)       # (width, height)
CNN_INPUT_SIZE = (64, 64)          # standard square input for CNN and crops

# Segmentation parameters
SEGMENTATION_MARGIN = 15           # margin around bounding box in pixels
MIN_CONTOUR_AREA = 400             # minimum area to avoid small noise specks

# Compute device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure necessary directories exist
for directory in [DATA_DIR, MASKS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "exploration"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)

