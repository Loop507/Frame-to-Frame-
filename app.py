import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import random

st.set_page_config(page_title="🌀 Transizione Video", layout="centered")

# === Effetti semplici e funzionanti ===

def fade(img1, img2, steps):
    return [(img1 * (1 - a) + img2 * a).astype(np.uint8) for a in np.linspace(0, 1, steps)]

def load_image(uploaded, size):
    img = Image.open(uploaded).convert("RGB").resize(size)
    return np.array(img)

def save_video(frames, path, fps):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

# === UI ===

st.title("🎬 Video da Immagini")
files = st.file_uploader("Carica almeno 2 immagini", type=["jpg", "png"], accept_multiple_files=True)
fps = st.slider("FPS", 10, 60, 30)
steps = st.slider("Frames per transizione", 10, 60, 20)

if st.button("🎥 Genera Video") and files and len(files) >= 2:
    st.info("🎞 Generazione in corso...")

    size = (640, 480)
    images = [load_image(f, size) for f in files]
    all_frames = []

    for i in range(len(images) - 1):
        trans = fade(images[i], images[i+1], steps)
        all_frames.extend(trans)

    # Loop finale opzionale
    trans = fade(images[-1], images[0], steps)
    all_frames.extend(trans)

    if not all_frames:
        st.error("❌ Nessun frame generato. Controlla le immagini.")
    else:
        out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        save_video(all_frames, out_path, fps)
        st.success("✅ Video generato!")
        st.video(out_path)
