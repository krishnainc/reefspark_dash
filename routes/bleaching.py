"""
Coral Bleaching Analysis routes
Provides HSV colour-based bleaching detection with heatmap output.
ML model slot is reserved — swap analyze_bleaching_model() in when ready.
"""
from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
import os
import uuid

bleaching_api = Blueprint('bleaching_api', __name__)

UPLOAD_FOLDER = "uploads/bleaching"
OUTPUT_FOLDER = "outputs/bleaching"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CORE ANALYSIS — HSV Colour Thresholding (v4, locked)
# ─────────────────────────────────────────────────────────────

def analyze_bleaching_hsv(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Crop out quadrat frame border (~6% each side)
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.06)
    img = img[margin:h - margin, margin:w - margin]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Dark substrate mask — exclude from both masks
    dark_mask = cv2.inRange(hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 55])
    )

    # Bleached — white / pale / lavender / grey tissue
    bleached_mask = cv2.inRange(hsv,
        np.array([0,  0,  130]),
        np.array([180, 55, 255])
    )

    # Healthy group 1 — colourful saturated tissue
    healthy_saturated = cv2.inRange(hsv,
        np.array([0,  80, 80]),
        np.array([180, 255, 185])
    )

    # Healthy group 2 — brown coral tissue (brain corals, hue 8–28)
    healthy_brown = cv2.inRange(hsv,
        np.array([8,  30, 55]),
        np.array([28, 130, 140])
    )

    healthy_mask = cv2.bitwise_or(healthy_saturated, healthy_brown)

    # Remove substrate from both masks
    bleached_mask = cv2.bitwise_and(bleached_mask, cv2.bitwise_not(dark_mask))
    healthy_mask  = cv2.bitwise_and(healthy_mask,  cv2.bitwise_not(dark_mask))

    # If pixel is in both, trust healthy over bleached
    bleached_mask = cv2.bitwise_and(bleached_mask, cv2.bitwise_not(healthy_mask))

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    bleached_mask = cv2.morphologyEx(bleached_mask, cv2.MORPH_OPEN, kernel)
    healthy_mask  = cv2.morphologyEx(healthy_mask,  cv2.MORPH_OPEN, kernel)

    bleached_px = cv2.countNonZero(bleached_mask)
    healthy_px  = cv2.countNonZero(healthy_mask)
    total       = bleached_px + healthy_px
    bleach_pct  = round((bleached_px / total * 100) if total > 0 else 0, 1)

    return {
        "bleach_pct":    bleach_pct,
        "bleached_mask": bleached_mask,
        "healthy_mask":  healthy_mask,
        "cropped_img":   img
    }


# ─────────────────────────────────────────────────────────────
# RESERVED — ML Model Slot (swap in after training)
# ─────────────────────────────────────────────────────────────

def analyze_bleaching_model(image_path):
    """
    TODO: Replace with YOLOv8 segmentation inference once model is trained.
    Expected return shape is identical to analyze_bleaching_hsv().
    """
    raise NotImplementedError("Coming soon")


# ─────────────────────────────────────────────────────────────
# HEATMAP GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_heatmap(result, output_path):
    img     = result["cropped_img"]
    overlay = img.copy()
    overlay[result["bleached_mask"] > 0] = [0, 0, 220]   # red  = bleached
    overlay[result["healthy_mask"]  > 0] = [0, 180, 0]   # green = healthy
    heatmap = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    cv2.imwrite(output_path, heatmap)


# ─────────────────────────────────────────────────────────────
# VERDICT LOGIC
# ─────────────────────────────────────────────────────────────

def get_bleaching_verdict(bleach_pct, method):
    disclaimer = (
        "Score estimated using colour analysis only. "
        "Substrate and shadows may affect accuracy by ±20–30%. "
        "Premium ML analysis coming soon."
    ) if method == "hsv" else ""

    if bleach_pct <= 10:
        return {
            "status":         "healthy",
            "verdict":        "Coral is healthy, no intervention needed",
            "saveable":       True,
            "recommendation": "Monitor regularly. No immediate action required.",
            "disclaimer":     disclaimer
        }
    elif bleach_pct <= 30:
        return {
            "status":         "mild",
            "verdict":        "Mild bleaching, saveable with monitoring",
            "saveable":       True,
            "recommendation": "Flag for monitoring. Recovery likely if stressor removed within 4–6 weeks.",
            "disclaimer":     disclaimer
        }
    elif bleach_pct <= 50:
        return {
            "status":         "moderate",
            "verdict":        "Moderate bleaching, at risk, may recover",
            "saveable":       True,
            "recommendation": "Prioritise for intervention. Consider shading or water quality measures. Time-sensitive.",
            "disclaimer":     disclaimer
        }
    elif bleach_pct <= 75:
        return {
            "status":         "severe",
            "verdict":        "Severe bleaching, unlikely to recover fully",
            "saveable":       False,
            "recommendation": "Collect tissue samples for coral nursery if possible. Report to reef authority.",
            "disclaimer":     disclaimer
        }
    else:
        return {
            "status":         "critical",
            "verdict":        "Critical bleaching, coral presumed lost",
            "saveable":       False,
            "recommendation": "Document for mortality tracking. Focus resources on surrounding healthy corals.",
            "disclaimer":     disclaimer
        }


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@bleaching_api.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file   = request.files['image']
    method = request.form.get('method', 'hsv')  # 'hsv' or 'model'

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save upload
    uid        = str(uuid.uuid4())[:8]
    input_path  = os.path.join(UPLOAD_FOLDER, f"{uid}_input.jpg")
    output_path = os.path.join(OUTPUT_FOLDER, f"{uid}_heatmap.jpg")
    file.save(input_path)

    # Run analysis
    try:
        if method == 'model':
            result = analyze_bleaching_model(input_path)
        else:
            result = analyze_bleaching_hsv(input_path)
    except NotImplementedError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    if result is None:
        return jsonify({'error': 'Could not read image — check file format'}), 400

    # Generate heatmap
    generate_heatmap(result, output_path)

    # Build response
    verdict = get_bleaching_verdict(result['bleach_pct'], method)

    return jsonify({
        'bleach_pct':   result['bleach_pct'],
        'heatmap_url':  f'/result/{uid}',
        'method':       method,
        **verdict
    })


@bleaching_api.route('/result/<uid>')
def get_result(uid):
    path = os.path.join(OUTPUT_FOLDER, f"{uid}_heatmap.jpg")
    if not os.path.exists(path):
        return jsonify({'error': 'Result not found'}), 404
    return send_file(path, mimetype='image/jpeg')