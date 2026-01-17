import os, cv2

VIDEO_PATH = "trip.mp4"         # GANTI ke nama video kamu
OUT_DIR = "input_images_5fps"
FPS_OUT = 5
MAX_WIDTH = 1280                # 0 kalau tidak mau resize

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

step = max(1, int(round(fps / FPS_OUT)))

i = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if i % step == 0:
        if MAX_WIDTH and frame.shape[1] > MAX_WIDTH:
            h, w = frame.shape[:2]
            new_h = int(h * MAX_WIDTH / w)
            frame = cv2.resize(frame, (MAX_WIDTH, new_h))

        ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        out = os.path.join(OUT_DIR, f"frame_{saved:06d}_{ms}.jpg")
        cv2.imwrite(out, frame)
        saved += 1

    i += 1

cap.release()
print("Saved frames:", saved)
