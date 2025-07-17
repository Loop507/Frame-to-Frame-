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

def noise_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frames.append(frame)
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [fade_effect, morph_effect, glitch_effect, noise_effect]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)

# --- INTERFACCIA STREAMLIT ---
st.title("🎞️ Frame-to-Frame FX Video Generator")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica le immagini (min. 2)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

fps = st.slider("Frame rate (FPS)", 5, 60, 24)
duration = st.slider("Durata totale video (secondi)", 1, 20, 10)
effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Noise", "Random"])
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

    total_frames = fps * duration
    transitions = len(images) - 1
    frames_per_transition = total_frames // transitions
    all_frames = []

    st.info("🎬 Generazione video in corso...")
    progress = st.progress(0)

    for i in range(transitions):
        img1, img2 = images[i], images[i + 1]

        if effect_choice == "Fade":
            frames = fade_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Morph":
            frames = morph_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Glitch":
            frames = glitch_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Noise":
            frames = noise_effect(img1, img2, frames_per_transition)
        else:
            frames = random_effect(img1, img2, frames_per_transition)

        all_frames.extend(frames)
        progress.progress((i + 1) / transitions)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    writer = imageio.get_writer(filepath, fps=fps)
    for idx, frame in enumerate(tqdm(all_frames)):
        writer.append_data(frame)
        if idx % 10 == 0:
            progress.progress(idx / len(all_frames))
    writer.close()

    st.success("✅ Video generato con successo!")
    st.markdown(
        f'<video controls width="320" height="240"><source src="{filepath}" type="video/mp4"></video>',
        unsafe_allow_html=True
    )
else:
    st.warning("📷 Carica almeno due immagini per iniziare.")
