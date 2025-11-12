import cv2
import streamlit as st
import torch
import ultralytics
from PIL import Image
import os

# ✅ Izinkan class DetectionModel agar PyTorch bisa load YOLO di versi 2.6+
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

from ultralytics import YOLO

# ✅ Cek versi OpenCV (opsional, hanya untuk debug)
st.write("✅ OpenCV version:", cv2.__version__)

# ✅ Load model
model_path = "best.pt"  # ubah sesuai nama file model kamu
if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file not found at {model_path}")
else:
    model = YOLO(model_path)
    st.title("🚀 Deteksi Objek dengan YOLOv8")

    # ✅ Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar yang diupload
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

        # Jalankan deteksi YOLO
        results = model(image)

        # Tampilkan hasil
        for r in results:
            im_array = r.plot()  # hasil prediksi dalam array BGR
            im = Image.fromarray(im_array[..., ::-1])  # ubah ke RGB untuk PIL
            st.image(im, caption="🎯 Hasil Deteksi", use_column_width=True)

