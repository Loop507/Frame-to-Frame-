import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
import random

st.set_page_config(page_title="Frame-to-Frame FX Video Generator by Loop507", layout="wide")

# --- FUNZIONI EFFETTI ---

def fade_effect(img1, img2, num_frames, intensity=1):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # intensity modifica la velocità della dissolvenza (più alta = transizione più lunga)
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
            for alpha in np.linspace(0, 1, int(num_frames*intensity))]

def morph_effect(img1, img2, num_frames, intensity=1):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8)
            for alpha in np.linspace(0, 1, int(num_frames*intensity))]

def glitch_effect(img1, img2, num_frames, intensity=1):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    max_shift = int(20 * intensity)
    glitch_lines = int(10 * intensity)
    for alpha in np.linspace(0, 1, int(num_frames*intensity)):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(glitch_lines):
            y = random.randint(0, h - 1)
            frame[y:y+1, :, :] = np.roll(frame[y:y+1, :, :], random.randint(-max_shift, max_shift), axis=1)
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_block_effect(img1, img2, num_frames, intensity=1):
    # Blocchi pixel crescenti in dissolvenza
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    max_block_size = int(50 * intensity)
    for alpha in np.linspace(0, 1, int(num_frames*intensity)):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        block_size = max(1, int(max_block_size * alpha))
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = frame[y:y+block_size, x:x+block_size]
                if block.size == 0:
                    continue
                avg_color = block.mean(axis=(0,1)).astype(np.uint8)
                frame[y:y+block_size, x:x+block_size] = avg_color
        frames.append(frame)
    return frames

def vhs_effect(img1, img2, num_frames, intensity=1):
    # Effetto VHS: linee orizzontali distorte + leggera saturazione/rumore
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    max_shift = int(15 * intensity)
    noise_strength = int(15 * intensity)
    for alpha in np.linspace(0, 1, int(num_frames*intensity)):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for y in range(0, h, 4):
            shift = random.randint(-max_shift, max_shift)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], shift, axis=1)
        noise = np.random.randint(0, noise_strength, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frames.append(frame)
    return frames

def color_echo_effect(img1, img2, num_frames, intensity=1):
    # Effetto eco colori spostati in orizzontale
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    frames = []
    max_shift = int(15 * intensity)
    for alpha in np.linspace(0, 1, int(num_frames*intensity)):
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # Splitting channels
        b, g, r = cv2.split(blended)
        shift = int(max_shift * alpha)
        b_shifted = np.roll(b, shift, axis=1)
        r_shifted = np.roll(r, -shift, axis=1)
        merged = cv2.merge([b_shifted, g, r_shifted])
        frames.append(merged)
    return frames

# Nuovo effetto 1: Zoom Morph (zoom progressivo tra img1 e img2)
def zoom_morph_effect(img1, img2, num_frames, intensity=1):
    img1 = np.array(img1).astype(np.uint32)
    img2 = np.array(img2).astype(np.uint32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for i, alpha in enumerate(np.linspace(0, 1, int(num_frames*intensity))):
        zoom_factor = 1 + 0.5 * alpha  # zoom from 1x to 1.5x
        center_x, center_y = w//2, h//2

        # zoom on img1 shrinking
        M1 = cv2.getRotationMatrix2D((center_x, center_y), 0, 1/zoom_factor)
        zoomed_img1 = cv2.warpAffine(img1, M1, (w, h))

        # zoom on img2 growing
        M2 = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
        zoomed_img2 = cv2.warpAffine(img2, M2, (w, h))

        blended = ((1 - alpha) * zoomed_img1 + alpha * zoomed_img2).astype(np.uint8)
        frames.append(blended)
    return frames

# Nuovo effetto 2: Wave Distort (onde orizzontali di distorsione)
def wave_distort_effect(img1, img2, num_frames, intensity=1):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    max_amplitude = 15 * intensity
    for alpha in np.linspace(0, 1, int(num_frames*intensity)):
        blended = img1 * (1 - alpha) + img2 * alpha
        frame = np.zeros_like(blended)
        for y in range(h):
            shift = int(max_amplitude * np.sin(2 * np.pi * (y / 30.0 + alpha * 5)))
            frame[y] = np.roll(blended[y], shift, axis=0)
        frames.append(frame.astype(np.uint8))
    return frames

# Effetto random aggiornato
def random_effect(img1, img2, num_frames, intensity=1):
    effect_list = [
        fade_effect,
        morph_effect,
        glitch_effect,
        pixel_block_effect,
        vhs_effect,
        color_echo_effect,
        zoom_morph_effect,
        wave_distort_effect,
    ]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames, intensity)

# --- INTERFACCIA STREAMLIT ---

st.title("Frame-to-Frame FX Video Generator by Loop507")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica le immagini (min. 2, max 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

num_seconds = st.number_input("Durata totale video (secondi)", min_value=1, max_value=60, value=10, step=1)
effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Pixel Block", "VHS", "Color Echo", "Zoom Morph", "Wave Distort", "Random"])
effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])

# Mappa intensità effetti
strength_map = {
    "Soft": 0.6,
    "Medio": 1.0,
    "Hard": 1.5,
}

if uploaded_files and 2 <= len(uploaded_files) <= 10:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    if output_format == "1:1":
        target_size = (512, 512)
    elif output_format == "9:16":
        target_size = (540, 960)
    else:
        target_size = (960, 540)

    images = [img.resize(target_size) for img in images]

    # Calcolo frame totali (fps 24)
    fps = 24
    total_frames = num_seconds * fps
    # Frame per transizione
    transitions = len(images) - 1
    frames_per_transition = total_frames // transitions

    st.info(f"Video durerà circa {num_seconds}s con {transitions} transizioni, {frames_per_transition} frame ciascuna")

    all_frames = []
    intensity_val = strength_map[effect_strength]

    # Funzioni mappate
    effect_funcs = {
        "Fade": fade_effect,
        "Morph": morph_effect,
        "Glitch": glitch_effect,
        "Pixel Block": pixel_block_effect,
        "VHS": vhs_effect,
        "Color Echo": color_echo_effect,
        "Zoom Morph": zoom_morph_effect,
        "Wave Distort": wave_distort_effect,
        "Random": random_effect,
    }

    generate_btn = st.button("Genera Video")

    if generate_btn:
        progress = st.progress(0)
        status_text = st.empty()

        for i in range(transitions):
            img1, img2 = images[i], images[i+1]
            frames = effect_funcs[effect_choice](img1, img2, frames_per_transition, intensity_val)
            all_frames.extend(frames)
            progress.progress((i + 1) / transitions)
            status_text.text(f"Generazione transizione {i + 1} di {transitions}...")

        # Salva video temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            filepath = tmpfile.name

        try:
            writer = imageio.get_writer(filepath, fps=fps, codec="libx264", quality=8)
            for frame in all_frames:
                writer.append_data(frame)
            writer.close()
            st.success("✅ Video generato con successo!")
            st.download_button(label="📥 Scarica Video", data=open(filepath, "rb").read(), file_name="video.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Errore durante la generazione del video: {e}")
else:
    if uploaded_files:
        st.warning("Carica da 2 a 10 immagini per generare il video.")

