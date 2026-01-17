import os
import glob
import torch
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =====================
# CONFIG
# =====================
INPUT_DIR = "input_images_5fps"
OUT_BOX_DIR = "output_boxes_5fps"
OUT_LABEL_DIR = "output_labels_5fps"

PROMPT = "pothole on road"
CONF_THRESH = 0.35          # turunkan kalau ingin lebih banyak terdeteksi
CLASS_ID = 0                # pothole = class 0 (YOLO)

os.makedirs(OUT_BOX_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# =====================
# LOAD MODEL
# =====================
model = build_sam3_image_model(
    checkpoint_path="checkpoints/sam3.pt",
    device="cuda",
)

processor = Sam3Processor(
    model=model,
    device="cuda",
    confidence_threshold=CONF_THRESH
)

# =====================
# HELPERS
# =====================
def to_yolo_format(box_xyxy, W, H):
    """
    box_xyxy = [x1, y1, x2, y2] in pixel coords
    returns (x_center, y_center, w, h) normalized [0..1]
    """
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(x1), W - 1))
    x2 = max(0.0, min(float(x2), W - 1))
    y1 = max(0.0, min(float(y1), H - 1))
    y2 = max(0.0, min(float(y2), H - 1))

    # ensure correct ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = (x2 - x1)
    bh = (y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return (xc / W, yc / H, bw / W, bh / H)

# =====================
# COLLECT IMAGES
# =====================
image_paths = sorted(
    glob.glob(f"{INPUT_DIR}/*.jpg") +
    glob.glob(f"{INPUT_DIR}/*.jpeg") +
    glob.glob(f"{INPUT_DIR}/*.png")
)

print(f"Found {len(image_paths)} images")

for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    overlay_path = f"{OUT_BOX_DIR}/{base}_box.jpg"
    label_path = f"{OUT_LABEL_DIR}/{base}.txt"

    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    # default: no detection -> empty label file
    best_box = None
    best_score = None

    state = {}
    state = processor.set_image(image, state)
    state = processor.set_text_prompt(PROMPT, state)

    boxes = state.get("boxes", None)   # [N, 4] xyxy
    scores = state.get("scores", None)

    if boxes is not None and scores is not None and len(scores) > 0:
        best_idx = scores.argmax().item()
        best_score = float(scores[best_idx])
        if best_score >= CONF_THRESH:
            best_box = boxes[best_idx].detach().cpu().tolist()

    # --- write YOLO label (always create file) ---
    with open(label_path, "w") as f:
        if best_box is not None:
            x, y, w, h = to_yolo_format(best_box, W, H)
            f.write(f"{CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # --- draw overlay (always create image) ---
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.text((10, 10), f"DETECTED score={best_score:.3f}", fill=(255, 0, 0))
        print(f"[OK] {base}: score={best_score:.3f} -> label+overlay")
    else:
        draw.text((10, 10), "NO DETECTION", fill=(255, 0, 0))
        print(f"[--] {base}: no detection -> empty label, overlay saved")

    overlay.save(overlay_path)

print("\nDONE")
print(f"Overlay images  : {OUT_BOX_DIR}")
print(f"YOLO label files: {OUT_LABEL_DIR}")
