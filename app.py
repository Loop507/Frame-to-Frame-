import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
import random
from tqdm import tqdm

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Video Generator by Loop507", layout="wide")

# --- EFFETTI ---
def fade_effect(img1, img2, num_frames):
    img1, img2 = np.array(img1).astype(np.float32), np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

def morph_effect(img1, img2, num_frames):
    img1, img2 = np.array(img1), np.array(img2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

def glitch_effect(img1, img2, num_frames):
    img1, img2 = np.array(img1), np.array(img2)
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

def pixel_block_effect(img1, img2, num_frames):
    img1, img2 = np.array(img1), np.array(img2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(30):
            x, y = random.randint(0, w-8), random.randint(0, h-8)
            blended[y:y+4, x:x+4] = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        frames.append(blended)
    return frames

def line_noise_effect(img1, img2, num_frames):
    img1, img2 = np.array(img1), np.array(img2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(10):
            y = random.randint(0, h-1)
            cv2.line(blended, (0, y), (w, y), (random.randint(0,255),)*3, 1)
        frames.append(blended)
    return frames

def color_echo_effect(img1, img2, num_frames):
    img1, img2 = np.array(img1), np.array(img2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        blend = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        b, g, r = cv2.split(blend)
        g = np.roll(g, 5, axis=1)
        r = np.roll(r, -5, axis=0)
        frames.append(cv2.merge([b, g, r]))
    return frames

def random_effect(img1, img2, num_frames):
    effects = [fade_effect, morph_effect, glitch_effect, pixel_block_effect, line_noise_effect, color_echo_effect]
    return random.choice(effects)(img1, img2, num_frames)

# --- INTERFACCIA STREAMLIT ---
st.title("🎞️ Frame-to-Frame FX Video Generator by Loop507")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica le immagini (min. 2)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

duration = st.slider("Durata totale del video (secondi)", 1, 30, 10)
effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Pixel", "Linee", "Eco Colori", "Random"])
effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])

if uploaded_files and len(uploaded_files) >= 2:
    if st.button("🎬 Genera Video"):
        images = [Image.open(file).convert("RGB") for file in uploaded_files]
        target_size = (512, 512) if output_format == "1:1" else (540, 960) if output_format == "9:16" else (960, 540)
        images = [img.resize(target_size) for img in images]

        strength_map = {"Soft": 10, "Medio": 20, "Hard": 40}
        frames_per_transition = strength_map[effect_strength]

        total_transitions = len(images) - 1
        total_frames = duration * 24
        frames_per_transition = max(5, total_frames // total_transitions)

        all_frames = []
        progress = st.progress(0.0, text="Generazione video in corso...")

        for i in range(total_transitions):
            img1, img2 = images[i], images[i+1]
            if effect_choice == "Fade":
                frames = fade_effect(img1, img2, frames_per_transition)
            elif effect_choice == "Morph":
                frames = morph_effect(img1, img2, frames_per_transition)
            elif effect_choice == "Glitch":
                frames = glitch_effect(img1, img2, frames_per_transition)
            elif effect_choice == "Pixel":
                frames = pixel_block_effect(img1, img2, frames_per_transition)
            elif effect_choice == "Linee":
                frames = line_noise_effect(img1, img2, frames_per_transition)
            elif effect_choice == "Eco Colori":
                frames = color_echo_effect(img1, img2, frames_per_transition)
            else:
                frames = random_effect(img1, img2, frames_per_transition)
            all_frames.extend(frames)
            progress.progress((i+1)/total_transitions)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            filepath = tmpfile.name

        writer = imageio.get_writer(filepath, fps=24, codec='libx264')
        for frame in tqdm(all_frames):
            writer.append_data(frame)
        writer.close()

        st.success("✅ Video generato con successo!")
        with open(filepath, "rb") as f:
            st.download_button("📥 Scarica Video", f, file_name="frame_to_frame_output.mp4")
else:
    st.warning("Carica almeno due immagini per iniziare.")
