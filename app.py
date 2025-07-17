import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import random

st.set_page_config(page_title="🎞️ Frame-to-Frame FX", layout="wide")

# --- EFFETTI ---
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

def pixel_blocks_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    block_size = 20
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if random.random() < 0.1:
                    dx = random.randint(-block_size//2, block_size//2)
                    dy = random.randint(-block_size//2, block_size//2)
                    block = frame[y:y+block_size, x:x+block_size].copy()
                    ny = min(max(0, y+dy), h-block_size)
                    nx = min(max(0, x+dx), w-block_size)
                    frame[ny:ny+block_size, nx:nx+block_size] = block
        frames.append(frame.astype(np.uint8))
    return frames

def vhs_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # VHS-like scanlines
        for y in range(0, h, 2):
            frame[y,:,:] = (frame[y,:,:] * 0.6).astype(np.uint8)
        # RGB shift
        shift = int(5 * alpha)
        b, g, r = cv2.split(frame)
        b = np.roll(b, shift, axis=1)
        r = np.roll(r, -shift, axis=1)
        frame = cv2.merge((b, g, r))
        frames.append(frame.astype(np.uint8))
    return frames

def corrupted_lines_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        num_lines = int(10 * alpha)
        for _ in range(num_lines):
            y = random.randint(0, h-1)
            length = random.randint(20, w//2)
            start_x = random.randint(0, w - length)
            frame[y, start_x:start_x+length, :] = np.random.randint(0, 256, (length, 3))
        frames.append(frame.astype(np.uint8))
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [
        fade_effect, morph_effect, glitch_effect,
        pixel_blocks_effect, vhs_effect, corrupted_lines_effect
    ]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)

# --- INTERFACCIA ---
st.title("🎞️ Frame-to-Frame FX Video Generator")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "Carica immagini (minimo 2, massimo 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

video_duration = st.slider("Durata video (secondi)", min_value=3, max_value=30, value=10)
fps = 24

effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Pixel Blocks", "VHS", "Linee Corrotte", "Random"])
effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])

if uploaded_files and 2 <= len(uploaded_files) <= 10:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    # Risoluzione in base al formato
    if output_format == "1:1":
        target_size = (512, 512)
    elif output_format == "9:16":
        target_size = (540, 960)
    else:
        target_size = (960, 540)

    images = [img.resize(target_size) for img in images]

    # Calcolo numero frame totali e per transizione
    total_frames = video_duration * fps
    transitions = len(images) - 1
    frames_per_transition = max(total_frames // transitions, 1)

    # Mappa intensità (es. quantità di frame o forza effetti)
    strength_map = {"Soft": frames_per_transition // 3, "Medio": frames_per_transition // 2, "Hard": frames_per_transition}

    n_frames = strength_map[effect_strength]

    if st.button("Genera Video"):
        all_frames = []
        progress = st.progress(0)
        for i in range(transitions):
            img1, img2 = images[i], images[i+1]

            if effect_choice == "Fade":
                frames = fade_effect(img1, img2, n_frames)
            elif effect_choice == "Morph":
                frames = morph_effect(img1, img2, n_frames)
            elif effect_choice == "Glitch":
                frames = glitch_effect(img1, img2, n_frames)
            elif effect_choice == "Pixel Blocks":
                frames = pixel_blocks_effect(img1, img2, n_frames)
            elif effect_choice == "VHS":
                frames = vhs_effect(img1, img2, n_frames)
            elif effect_choice == "Linee Corrotte":
                frames = corrupted_lines_effect(img1, img2, n_frames)
            else:
                frames = random_effect(img1, img2, n_frames)

            all_frames.extend(frames)
            progress.progress((i+1)/transitions)

        # Scrittura video con OpenCV (più affidabile di imageio per mp4)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            video_path = tmpfile.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, target_size)

        for frame in all_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        st.success("✅ Video generato!")
        st.video(video_path)

else:
    st.warning("Carica almeno 2 immagini e massimo 10.")

