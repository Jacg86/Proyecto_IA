"""
HSV and OpenCV-based segmentation module for LEGO bricks.
Implements color space transformations, S and V channel thresholding (Otsu & adaptive),
morphological cleaning, contour detection, and Watershed object separation.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class HSVSegmenter:
    def __init__(self, 
                 min_area=config.MIN_CONTOUR_AREA, 
                 margin=config.SEGMENTATION_MARGIN,
                 dark_v_thresh=130):
        self.min_area = min_area
        self.margin = margin
        self.dark_v_thresh = dark_v_thresh
        
        # Load background reference images if available
        self.bg_a = cv2.imread(config.BG_BACKLIT_A) if os.path.exists(config.BG_BACKLIT_A) else None
        self.bg_b = cv2.imread(config.BG_BACKLIT_B) if os.path.exists(config.BG_BACKLIT_B) else None

    def segment_hsv_otsu(self, img):
        """
        Segment using HSV color space and Otsu thresholding:
        - S channel detects saturated colored pieces (red, blue, green, brown, yellow).
        - V channel inverted detects dark and black pieces.
        - Logical OR combines both channels.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # 1. Mask for colored pieces via S channel Otsu
        _, mask_s = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Mask for dark / black pieces via V channel
        # Safe inverted V threshold to catch dark pieces without catching background variations
        _, mask_v = cv2.threshold(v_channel, self.dark_v_thresh, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Combine with bitwise OR
        mask_combined = cv2.bitwise_or(mask_s, mask_v)
        
        # Suppress camera/fixture border margins (15 pixels around perimeter)
        h_i, w_i = img.shape[:2]
        inner_mask = np.zeros((h_i, w_i), dtype=np.uint8)
        inner_mask[12:h_i-12, 12:w_i-12] = 255
        mask_combined = cv2.bitwise_and(mask_combined, inner_mask)
        
        return mask_combined

    def segment_hsv_adaptive(self, img):
        """
        Enhanced HSV segmentation for challenging cases (such as white or light pieces):
        Combines S-channel Otsu, dark V detection, and background color hue exclusion.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # S high threshold
        _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dark pieces
        _, mask_v = cv2.threshold(v, self.dark_v_thresh, 255, cv2.THRESH_BINARY_INV)
        
        # Backlit table is light cyan: Hue ~92-120, Sat ~40-130, Val > 170
        bg_cyan_mask = cv2.inRange(hsv, np.array([92, 40, 160]), np.array([125, 135, 255]))
        fg_hue_contrast = cv2.bitwise_not(bg_cyan_mask)
        
        # Local edge gradient for subtle borders (Canny on grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred_gray, 30, 90)
        edge_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        combined = cv2.bitwise_or(mask_s, mask_v)
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(fg_hue_contrast, edge_dilated))
        
        # Suppress borders
        h_i, w_i = img.shape[:2]
        inner_mask = np.zeros((h_i, w_i), dtype=np.uint8)
        inner_mask[12:h_i-12, 12:w_i-12] = 255
        combined = cv2.bitwise_and(combined, inner_mask)
        
        return combined

    def segment_background_subtraction(self, img):
        """
        Segmentation using difference with reference backlit background.
        Used as comparative baseline.
        """
        if self.bg_a is None and self.bg_b is None:
            return self.segment_hsv_otsu(img)
            
        bg = self.bg_b if self.bg_b is not None else self.bg_a
        diff = cv2.absdiff(img, bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        
        h_i, w_i = img.shape[:2]
        inner_mask = np.zeros((h_i, w_i), dtype=np.uint8)
        inner_mask[12:h_i-12, 12:w_i-12] = 255
        mask = cv2.bitwise_and(mask, inner_mask)
        return mask

    def segment_universal_adaptive(self, img):
        """
        Universal adaptive segmentation for external images taken under arbitrary
        ambient lighting or on plain non-backlit backgrounds (e.g. paper or desk):
        - Applies CLAHE contrast equalization
        - Adaptive Gaussian local thresholding
        - Combined with S-channel saturation detection and morphological filling.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)
        
        # Adaptive thresholding detects local contrast edges regardless of illumination gradients
        thresh_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
        )
        
        # Color saturation boost
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        _, s_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)
        
        combined = cv2.bitwise_or(thresh_adapt, s_mask)
        
        # Morphological closing with elliptical kernel to fill internal contours
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=3)
        
        # Border suppression
        h_i, w_i = img.shape[:2]
        inner_mask = np.zeros((h_i, w_i), dtype=np.uint8)
        inner_mask[12:h_i-12, 12:w_i-12] = 255
        return cv2.bitwise_and(closed, inner_mask)

    def clean_mask(self, mask, open_size=5, close_size=9, close_iters=2):
        """
        Applies morphological operations:
        - Opening: eliminates small noise specks and isolated background pixels.
        - Closing: fills small holes and studs inside the LEGO brick.
        """
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=close_iters)
        return closed

    def apply_watershed_separation(self, img, binary_mask):
        """
        Evaluates Watershed segmentation for separating touching or clustered LEGO bricks.
        Returns a refined binary mask with separated objects.
        """
        # Sure background area by dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
        
        # Sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.45 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown boundary region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        # Add 1 to all labels so background is 1 instead of 0
        markers = markers + 1
        # Mark unknown region as 0
        markers[unknown == 255] = 0
        
        # Run Watershed
        markers_out = cv2.watershed(img.copy(), markers)
        
        # Resulting mask without boundaries (-1) and without background (1)
        separated_mask = np.zeros_like(binary_mask)
        separated_mask[markers_out > 1] = 255
        return separated_mask

    def remove_skin_tones(self, img, mask):
        """
        Suppresses human skin pixels (e.g. hands or fingers holding LEGO pieces)
        using YCrCb color space thresholding.
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # Standard human skin range in YCrCb: Cr in [133, 173], Cb in [77, 127]
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        # Dilate slightly to catch finger boundaries
        skin_dilated = cv2.dilate(skin_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        return cv2.bitwise_and(mask, cv2.bitwise_not(skin_dilated))

    def detect_and_extract_pieces(self, img, method='hsv_otsu', use_watershed=False, filter_skin=False):
        """
        Executes end-to-end segmentation:
        1. Segment image to binary mask
        2. Clean mask with morphological operations
        3. Optional Watershed object separation
        4. Detect contours and filter by minimum area
        5. Extract bounding boxes with configurable padding margin
        
        Returns:
            clean_mask: binary mask (0 or 255)
            pieces: list of dicts with bounding boxes, crops, contours, and areas.
        """
        if method == 'hsv_otsu':
            raw_mask = self.segment_hsv_otsu(img)
        elif method == 'hsv_adaptive':
            raw_mask = self.segment_hsv_adaptive(img)
        elif method == 'universal_adaptive':
            raw_mask = self.segment_universal_adaptive(img)
        elif method == 'bg_sub':
            raw_mask = self.segment_background_subtraction(img)
        else:
            raw_mask = self.segment_hsv_otsu(img)
            
        if filter_skin:
            raw_mask = self.remove_skin_tones(img, raw_mask)
            
        mask = self.clean_mask(raw_mask)
        
        if use_watershed:
            mask = self.apply_watershed_separation(img, mask)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = img.shape[:2]
        pieces = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Apply configurable margin
            x1 = max(0, x - self.margin)
            y1 = max(0, y - self.margin)
            x2 = min(w_img, x + w + self.margin)
            y2 = min(h_img, y + h + self.margin)
            
            crop = img[y1:y2, x1:x2].copy()
            piece_mask = mask[y1:y2, x1:x2].copy()
            
            pieces.append({
                "contour": cnt,
                "area": area,
                "tight_bbox": (x, y, w, h),
                "padded_bbox": (x1, y1, x2 - x1, y2 - y1),
                "crop": crop,
                "piece_mask": piece_mask
            })
            
        # Sort pieces by area descending
        pieces.sort(key=lambda p: p["area"], reverse=True)
        return mask, pieces

    def draw_detections(self, img, pieces, labels=None, confidences=None):
        """
        Draws contours, bounding boxes, and prediction labels onto the image.
        """
        overlay = img.copy()
        
        for idx, piece in enumerate(pieces):
            x, y, w, h = piece["padded_bbox"]
            cnt = piece["contour"]
            
            # Draw contour in bright green
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
            
            # Draw bounding box in cyan
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 200, 0), 2)
            
            # Text label
            label_text = f"Piece #{idx+1}"
            if labels and idx < len(labels):
                label_text = labels[idx]
                if confidences and idx < len(confidences):
                    label_text += f" ({confidences[idx]*100:.1f}%)"
                    
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(overlay, (x, max(0, y - 22)), (x + tw + 6, max(22, y)), (0, 0, 0), -1)
            cv2.putText(overlay, label_text, (x + 3, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            
        return overlay

