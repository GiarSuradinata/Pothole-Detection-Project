import os, glob, shutil

INPUT_IMG_DIR = "input_images_5fps"
INPUT_LBL_DIR = "output_labels_5fps"
OUT_DIR = "pothole_yolo_5fps"

# split time (ms) untuk VIDEO saja
TRAIN_END_MS = 7 * 60 * 1000 + 58 * 1000   # 7:58 = 478000 ms

def is_video_frame(fname: str) -> bool:
    # Video frames dari extractor kita selalu diawali 'frame_'
    return fname.startswith("frame_")

def get_ms_from_frame(fname: str):
    # frame_000123_456789.jpg -> ms = 456789
    try:
        return int(fname.split("_")[-1].split(".")[0])
    except:
        return None

# collect images
exts = ["*.jpg", "*.jpeg", "*.png"]
images = []
for e in exts:
    images += glob.glob(os.path.join(INPUT_IMG_DIR, e))

if not images:
    raise SystemExit("No images found")

# prepare output dirs
for sub in ["images/train","images/val","labels/train","labels/val"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

train_count = val_count = 0
photo_train = 0
bad_frame_names = 0

for img_path in sorted(images):
    fname = os.path.basename(img_path)
    base = os.path.splitext(fname)[0]
    lbl_path = os.path.join(INPUT_LBL_DIR, base + ".txt")

    if not is_video_frame(fname):
        # FORCE all non-frame_* images into TRAIN (foto ekstra)
        split = "train"
        photo_train += 1
    else:
        ms = get_ms_from_frame(fname)
        if ms is None:
            # fallback: kalau ada frame_ yang aneh namanya, tetap TRAIN
            split = "train"
            bad_frame_names += 1
        elif ms <= TRAIN_END_MS:
            split = "train"
        else:
            split = "val"

    out_img = os.path.join(OUT_DIR, "images", split, fname)
    out_lbl = os.path.join(OUT_DIR, "labels", split, base + ".txt")

    shutil.copy2(img_path, out_img)

    if os.path.exists(lbl_path):
        shutil.copy2(lbl_path, out_lbl)
    else:
        open(out_lbl, "w").close()

    if split == "train":
        train_count += 1
    else:
        val_count += 1

print("=== TIME-BASED YOLO DATASET READY (FORCE PHOTOS->TRAIN) ===")
print("Total images :", len(images))
print("Train images :", train_count)
print("Val images   :", val_count)
print("Extra photos in train:", photo_train)
print("Weird frame_* names forced to train:", bad_frame_names)
print("Output folder:", OUT_DIR)
