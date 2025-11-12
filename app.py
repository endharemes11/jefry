import os
import cv2
import torch
import ultralytics
import streamlit as st
from PIL import Image

# ✅ Izinkan PyTorch memuat class YOLO (DetectionModel) agar tidak error
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

from ultralytics import YOLO

st.title("🚀 Deteksi Objek dengan YOLOv8")
st.write("✅ OpenCV version:", cv2.__version__)

# ✅ Pastikan file model ada
model_path = "best.pt"  # ubah sesuai nama file model kamu
if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file tidak ditemukan di {model_path}")
else:
    try:
        # ✅ Load model dengan mode aman (handle PyTorch >=2.6)
        with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel]):
            model = YOLO(model_path)
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

    # ✅ Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang diunggah", use_column_width=True)

        # Jalankan deteksi YOLO
        results = model(image)

        # Tampilkan hasil
        for r in results:
            im_array = r.plot()  # hasil deteksi dalam array BGR
            im = Image.fromarray(im_array[..., ::-1])  # ubah ke RGB
            st.image(im, caption="🎯 Hasil Deteksi", use_column_width=True)
