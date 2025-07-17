# app.py - Frame-to-Frame FX con tutti gli effetti e controlli completi

import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import imageio
import tempfile
import random
from pathlib import Path
from tqdm import tqdm

# Configurazione pagina
st.set_page_config(page_title="🎞️ Frame-to-Frame FX", layout="wide")

# === FUNZIONI EFFETTI ===
def fade_effect(img1, img2, num_frames):
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

def slide_effect(img1, img2, num_frames):
    h, w, _ = img1.shape
    return [np.hstack((img1[:, int(w*i/num_frames):], img2[:, :int(w*i/num_frames)])) for i in range(num_frames)]

def glitch_effect(img1, img2, num_frames):
    h, w, _ = img1.shape
    return [cv2.remap(img1, np.float32(np.tile(np.arange(w), (h, 1))) + np.random.randint(-10, 10),
                      np.float32(np.tile(np.arange(h)[:, np.newaxis], (1, w))), cv2.INTER_LINEAR).astype(np.uint8)
            for _ in range(num_frames)]

def morph_effect(img1, img2, num_frames):
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

# === EFFETTI DISPONIBILI ===
effects = {
    "Fade": fade_effect,
    "Slide": slide_effect,
    "Glitch": glitch_effect,
    "Morph": morph_effect
}

def random_effect(img1, img2, num_frames):
    effect_list = list(effects.values())
    return random.choice(effect_list)(img1, img2, num_frames)

# === APP ===
st.title("🎞️ Frame-to-Frame FX")

uploaded_images = st.file_uploader("Carica almeno due immagini", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
effect_name = st.selectbox("Scegli l'effetto di transizione", ["Random"] + list(effects.keys()))
strength = st.slider("Intensità (numero frame)", 3, 60, 12)
output_format = st.selectbox("Formato Output", ["1:1", "9:16", "16:9"])

if uploaded_images and len(uploaded_images) >= 2:
    images = [Image.open(img).convert("RGB") for img in uploaded_images]

    # Resize immagini secondo formato scelto
    def resize_format(img):
        if output_format == "1:1": return img.resize((512, 512))
        elif output_format == "9:16": return img.resize((540, 960))
        elif output_format == "16:9": return img.resize((960, 540))

    resized_images = [resize_format(np.array(img)) for img in images]

    all_frames = []
    for i in range(len(resized_images)-1):
        img1, img2 = resized_images[i], resized_images[i+1]
        func = random_effect if effect_name == "Random" else effects[effect_name]
        all_frames.extend(func(img1, img2, strength))

    # Preview leggera
    st.image(all_frames[len(all_frames)//2], caption="Anteprima (frame intermedio)", use_container_width=True)

    # Output video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    st.info("Generazione video in corso...")
    progress_bar = st.progress(0)

    writer = imageio.get_writer(filepath, fps=24)
    for i, frame in enumerate(all_frames):
        writer.append_data(frame)
        progress_bar.progress((i+1)/len(all_frames))
    writer.close()

    st.success("Video generato con successo!")
    with open(filepath, "rb") as f:
        st.download_button("📥 Scarica il video", f, file_name="output.mp4", mime="video/mp4")
else:
    st.warning("Carica almeno due immagini per generare una transizione.")
