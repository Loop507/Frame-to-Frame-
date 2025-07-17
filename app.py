import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
import random
from tqdm import tqdm

st.set_page_config(page_title="🎞️ Frame-to-Frame FX", layout="wide")

# --- FUNZIONI DI EFFETTO ---
def fade_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
            for alpha in np.linspace(0, 1, num_frames)]

def morph_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8)
            for alpha in np.linspace(0, 1, num_frames)]

def glitch_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(10):
            y = random.randint(0, h - 1)
            frame[y:y+1, :, :] = np.roll(frame[y:y+1, :, :], random.randint(-20, 20), axis=1)
        frames.append(frame.astype(np.uint8))
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [fade_effect, morph_effect, glitch_effect]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)

# --- INTERFACCIA STREAMLIT ---
st.title("🎞️ Frame-to-Frame FX Video Generator")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica le immagini (min. 2)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

num_frames = st.slider("Numero di frame per transizione", 5, 60, 20)
effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Random"])
effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])

if uploaded_files and len(uploaded_files) >= 2:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    if output_format == "1:1":
        target_size = (512, 512)
    elif output_format == "9:16":
        target_size = (540, 960)
    else:
        target_size = (960, 540)

    images = [img.resize(target_size) for img in images]

    all_frames = []
    strength_map = {"Soft": 10, "Medio": 20, "Hard": 40}
    n_frames = strength_map[effect_strength]

    st.info("Generazione video in corso...")
    progress = st.progress(0)

    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i+1]

        if effect_choice == "Fade":
            frames = fade_effect(img1, img2, n_frames)
        elif effect_choice == "Morph":
            frames = morph_effect(img1, img2, n_frames)
        elif effect_choice == "Glitch":
            frames = glitch_effect(img1, img2, n_frames)
        else:
            frames = random_effect(img1, img2, n_frames)

        all_frames.extend(frames)
        progress.progress((i + 1) / (len(images) - 1))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    writer = imageio.get_writer(filepath, fps=24)
    for frame in tqdm(all_frames):
        writer.append_data(frame)
    writer.close()

    st.success("✅ Video generato con successo!")
    st.video(filepath)
else:
    st.warning("Carica almeno due immagini per iniziare.")
