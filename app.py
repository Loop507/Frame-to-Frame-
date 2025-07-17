import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import tempfile
import random
from tqdm import tqdm

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

def vhs_effect(img1, img2, num_frames):
    # VHS style glitch: distorsione e linee orizzontali
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # righe orizzontali casuali shiftate
        for _ in range(15):
            y = random.randint(0, h - 2)
            shift = random.randint(-15, 15)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], shift, axis=1)
        # aggiungi rumore
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_block_effect(img1, img2, num_frames):
    # Effetto pixel blocco
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    block_size = 10
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # blocchi pixel casuali shiftati
        for _ in range(15):
            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            shift_x = random.randint(-block_size, block_size)
            shift_y = random.randint(-block_size, block_size)
            block = frame[y:y+block_size, x:x+block_size].copy()
            frame[y+shift_y:y+shift_y+block_size, x+shift_x:x+shift_x+block_size] = block
        frames.append(frame.astype(np.uint8))
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [fade_effect, morph_effect, glitch_effect, vhs_effect, pixel_block_effect]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)

# --- INTERFACCIA STREAMLIT ---
st.title("🎞️ Frame-to-Frame FX Video Generator")

uploaded_files = st.file_uploader("Carica almeno 2 immagini", type=["png","jpg","jpeg"], accept_multiple_files=True)

output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

duration_sec = st.slider("Durata totale video (secondi)", 3, 30, 10)

effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "VHS", "Pixel Block", "Random"])

effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])

if uploaded_files and len(uploaded_files) >= 2:
    images = [Image.open(f).convert("RGB") for f in uploaded_files]

    # Calcola dimensione output
    if output_format == "1:1":
        target_size = (512, 512)
    elif output_format == "9:16":
        target_size = (540, 960)
    else:
        target_size = (960, 540)

    images = [img.resize(target_size) for img in images]

    # Calcola frames totali e frames per transizione
    fps = 24
    total_frames = duration_sec * fps
    n_transitions = len(images) - 1
    frames_per_transition = max(total_frames // n_transitions, 1)

    strength_map = {"Soft": 5, "Medio": 10, "Hard": 20}
    strength = strength_map[effect_strength]

    all_frames = []

    st.info("Generazione video in corso...")
    progress = st.progress(0)

    for i in range(n_transitions):
        img1, img2 = images[i], images[i+1]
        num_frames = min(frames_per_transition, strength)
        if effect_choice == "Fade":
            frames = fade_effect(img1, img2, num_frames)
        elif effect_choice == "Morph":
            frames = morph_effect(img1, img2, num_frames)
        elif effect_choice == "Glitch":
            frames = glitch_effect(img1, img2, num_frames)
        elif effect_choice == "VHS":
            frames = vhs_effect(img1, img2, num_frames)
        elif effect_choice == "Pixel Block":
            frames = pixel_block_effect(img1, img2, num_frames)
        else:
            frames = random_effect(img1, img2, num_frames)

        all_frames.extend(frames)
        progress.progress((i+1)/n_transitions)

    # Salva video temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    writer = imageio.get_writer(filepath, fps=fps, codec='libx264', format='mp4')
    for frame in tqdm(all_frames):
        writer.append_data(frame)
    writer.close()

    st.success("✅ Video generato con successo!")
    st.video(filepath, format="video/mp4", start_time=0, use_container_width=False)

else:
    st.warning("Carica almeno 2 immagini per iniziare.")
