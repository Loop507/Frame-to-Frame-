import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
import random

st.set_page_config(page_title="Frame-to-Frame FX Video Generator by Loop507", layout="wide")

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

def slide_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for i in range(num_frames):
        offset = int(w * i / num_frames)
        frame = np.zeros_like(img1)
        frame[:, :w - offset] = img1[:, offset:]
        frame[:, w - offset:] = img2[:, :offset]
        frames.append(frame)
    return frames

def invert_colors_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        inv_img1 = 255 - img1
        frame = cv2.addWeighted(inv_img1, 1 - alpha, img2, alpha, 0)
        frames.append(frame.astype(np.uint8))
    return frames

def color_echo_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    frames = []
    echo_intensity = 0.6
    prev_frame = img1.copy()
    for alpha in np.linspace(0, 1, num_frames):
        base = img1 * (1 - alpha) + img2 * alpha
        frame = base * (1 - echo_intensity) + prev_frame * echo_intensity
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
        prev_frame = frame.astype(np.float32)
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [fade_effect, morph_effect, glitch_effect, slide_effect, invert_colors_effect, color_echo_effect]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)

# --- INTERFACCIA STREAMLIT ---
st.markdown("""
    <h1 style='font-size: 28px;'>Frame-to-Frame FX Video Generator <span style='font-size:14px; color: gray;'>by Loop507</span></h1>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica le immagini (min. 2)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

num_frames = st.slider("Frame per transizione", 5, 60, 20)
effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Slide", "Invert Colors", "Eco Colori", "Random"])
effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])

# Durata video in secondi
duration_sec = st.number_input("Durata totale video (secondi)", min_value=1, max_value=60, value=10, step=1)

if uploaded_files and len(uploaded_files) >= 2:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    if output_format == "1:1":
        target_size = (512, 512)
    elif output_format == "9:16":
        target_size = (540, 960)
    else:
        target_size = (960, 540)

    images = [img.resize(target_size) for img in images]

    strength_map = {"Soft": 10, "Medio": 20, "Hard": 40}
    n_frames_per_transition = strength_map[effect_strength]

    # Calcolo fps per rispettare durata video totale
    total_transitions = len(images) - 1
    total_frames = total_transitions * n_frames_per_transition
    fps = total_frames / duration_sec

    # Bottone per generare video
    if st.button("Genera Video"):
        all_frames = []
        progress = st.progress(0)

        for i in range(total_transitions):
            img1, img2 = images[i], images[i + 1]

            if effect_choice == "Fade":
                frames = fade_effect(img1, img2, n_frames_per_transition)
            elif effect_choice == "Morph":
                frames = morph_effect(img1, img2, n_frames_per_transition)
            elif effect_choice == "Glitch":
                frames = glitch_effect(img1, img2, n_frames_per_transition)
            elif effect_choice == "Slide":
                frames = slide_effect(img1, img2, n_frames_per_transition)
            elif effect_choice == "Invert Colors":
                frames = invert_colors_effect(img1, img2, n_frames_per_transition)
            elif effect_choice == "Eco Colori":
                frames = color_echo_effect(img1, img2, n_frames_per_transition)
            else:
                frames = random_effect(img1, img2, n_frames_per_transition)

            all_frames.extend(frames)
            progress.progress((i + 1) / total_transitions)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            filepath = tmpfile.name

        writer = imageio.get_writer(filepath, fps=fps, codec="libx264")
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()

        st.success("✅ Video generato con successo!")
        st.download_button("⬇️ Scarica Video", filepath, file_name="frame_to_frame_fx.mp4", mime="video/mp4")

else:
    st.warning("Carica almeno due immagini per iniziare.")
