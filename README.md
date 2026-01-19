# Pothole-Detection-Project
Final Project for my Computer Vision Bootcamp 

🕳️ Sistem Deteksi Lubang Jalan (Pothole Detection)

Pelabelan Otomatis (SAM3) → YOLOv8 → Pipeline Deployment

📌 Gambaran Umum

Project ini merupakan Proof of Concept (PoC) untuk membangun sistem deteksi lubang jalan (pothole) secara end-to-end, dengan fokus utama pada:

arsitektur deployment

pelabelan data otomatis

pemrosesan video secara asynchronous (offline batch processing)

Sistem ini memungkinkan pengguna untuk mengunggah video perjalanan (MP4) dan data GPS (GPX), yang kemudian diproses pada server dengan GPU untuk menghasilkan hasil deteksi pothole beserta metadata lokasinya.

Project ini tidak berfokus pada real-time inference di perangkat mobile, melainkan pada alur sistem dan integrasi antar komponen.

🎯 Fokus Project

Fokus utama project ini adalah:

✅ Desain pipeline deployment
✅ Pelabelan otomatis menggunakan SAM3
✅ Pemrosesan video secara batch (offline)
✅ Sistem job-based & asynchronous

Optimasi performa model dilakukan secara iteratif dan bukan tujuan utama tahap PoC ini.

🧠 Kontribusi Utama

Pelabelan Otomatis dengan SAM3

Menghilangkan kebutuhan pelabelan bounding box secara manual

Mempercepat pembuatan dataset dari video mentah

Bertindak sebagai teacher model untuk melatih model deteksi ringan

Training YOLOv8 dari Data Berlabel Otomatis

Model ringan dan cocok untuk deployment

Seluruh dataset dihasilkan dari proses pelabelan otomatis

Arsitektur Deployment End-to-End

Frontend → API → Server GPU → Penyimpanan metadata

Dirancang untuk pemrosesan asynchronous dan scalable

🏗️ Arsitektur Sistem (Berorientasi Deployment)
User
  │
  ▼
Streamlit (Cloud UI)
  │
  ├── Upload MP4 + GPX
  │
  ▼
FastAPI Backend (via ngrok)
  │
  ├── Generate Job ID
  ├── Simpan metadata upload
  ▼
Laptop / Server GPU
  │
  ├── Parsing GPS
  ├── Ekstraksi frame video
  ├── Inferensi YOLO
  ├── Sinkronisasi frame & GPS
  ▼
Database Metadata
(Job ID, GPS, frame, hasil inferensi, status)
  │
  ▼
Agregasi Hasil
  │
  ▼
Streamlit UI (Visualisasi Hasil)


📌 Sistem ini berjalan dalam mode offline batch processing, bukan real-time.

🧪 Dataset & Strategi Pelabelan
Sumber Data

Video perjalanan kendaraan (~10 menit)

Tambahan ±51 foto pothole

Strategi Sampling

Ekstraksi frame pada 5 FPS

Setiap frame memiliki file label (kosong atau berisi)

Pelabelan Otomatis

Menggunakan SAM3

Output berupa bounding box format YOLO

Tidak dilakukan pelabelan manual

Ringkasan Dataset (5 FPS)
Item	Jumlah
Total frame	3045
Frame dengan pothole	355
Frame tanpa pothole	2690
Data train	2443
Data validasi	602
Instance pothole (validasi)	30

📌 Validasi menggunakan time-based split untuk mencegah data leakage antar frame yang berdekatan.

🤖 Model & Training
Model Deteksi

YOLOv8n (Ultralytics)

Konfigurasi Training

Resolusi input: 640

Epoch: 60 → 120

Dataset: hasil pelabelan otomatis SAM3

GPU: RTX 4090

OS: Ubuntu 22.04 (WSL2)

Model Terbaik
runs/detect/train4/weights/best.pt

📊 Hasil (Tahap PoC)
Metrik	Nilai
Precision	0.799
Recall	0.233
mAP@50	0.301
Interpretasi

Precision tinggi menunjukkan prediksi yang muncul relatif akurat

Recall masih terbatas, terutama pada pothole kecil atau ambigu

Keterbatasan ini wajar untuk PoC dengan pelabelan otomatis

🎥 Inferensi & Deployment

Inferensi dilakukan sebagai background job:

yolo detect predict \
  model=best.pt \
  source=video.mp4 \
  conf=0.10 \
  iou=0.4


Hasil inferensi:

disimpan berdasarkan Job ID

dikaitkan dengan data GPS

ditampilkan kembali ke frontend setelah proses selesai

⚠️ Keterbatasan Saat Ini

Pelabelan otomatis menghasilkan noise

Jumlah instance pada data validasi relatif kecil

Belum ada inferensi real-time di perangkat mobile

Belum ada pelacakan temporal antar frame

Keterbatasan ini diterima dan sesuai dengan tujuan PoC.

🚀 Pengembangan Selanjutnya

Human-in-the-loop relabeling

Training dengan resolusi lebih tinggi (imgsz=960)

Temporal tracking antar frame

Visualisasi berbasis peta (GeoJSON / Map)

Export model ke ONNX / TensorRT

📌 Status Project

🟡 Proof of Concept – Pipeline Deployment Berfungsi End-to-End


