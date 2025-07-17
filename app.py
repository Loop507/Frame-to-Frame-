import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import tempfile
import random
import os

st.set_page_config(page_title="🎞️ Frame-to-Frame FX", layout="wide")

# --- EFFETTI ---
def fade_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

def morph_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

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

def pixel_block_effect(img1, img2, num_frames):
    # Effetto blocchi pixel casuali durante la transizione
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    block_size = 16
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # Blocchi pixel random
        for _ in range(50):
            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            frame[y:y+block_size, x:x+block_size, :] = np.random.randint(0, 256, (block_size, block_size, 3), dtype=np.uint8)
        frames.append(frame.astype(np.uint8))
    return frames

def vhs_effect(img1, img2, num_frames):
    # VHS-style static + distorsioni
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        # Linee orizzontali statiche
        for y in range(0, h, 8):
            line_offset = random.randint(-10, 10)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], line_offset, axis=1)
        frames.append(frame.astype(np.uint8))
    return frames

def corrupted_lines_effect(img1, img2, num_frames):
    # Linee pixel corrotte verticali
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(20):
            x = random.randint(0, w - 1)
            frame[:, x:x+2, :] = np.roll(frame[:, x:x+2, :], random.randint(-20, 20), axis=0)
        frames.append(frame.astype(np.uint8))
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [
        fade_effect,
        morph_effect,
        glitch_effect,
        pixel_block_effect,
        vhs_effect,
        corrupted_lines_effect
    ]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)

# --- STREAMLIT UI ---
st.title("🎞️ Frame-to-Frame FX Video Generator")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica almeno 2 immagini", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

num_frames_per_transition = st.slider("Frame per transizione", 5, 60, 20)
effect_choice = st.selectbox("Effetto", ["Fade", "Morph", "Glitch", "Pixel Block", "VHS", "Corrupted Lines", "Random"])
effect_strength = st.selectbox("Intensità Effetto", ["Soft", "Medio", "Hard"])
video_duration = st.slider("Durata video (secondi)", 3, 30, 10)

if uploaded_files and len(uploaded_files) >= 2:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    # Imposta dimensione in base al formato scelto
    if output_format == "1:1":
        target_size = (512, 512)
    elif output_format == "9:16":
        target_size = (540, 960)
    else:
        target_size = (960, 540)

    images = [img.resize(target_size) for img in images]

    # Calcolo totale frame video = fps * durata
    fps = 24
    total_frames_video = fps * video_duration

    # Numero transizioni = immagini - 1
    total_transitions = len(images) - 1

    # Frames per transizione proporzionati a durata
    frames_per_effect = max(total_frames_video // total_transitions, 5)

    # Override intensità per variare frames per effetto
    strength_map = {"Soft": int(frames_per_effect * 0.6), "Medio": frames_per_effect, "Hard": int(frames_per_effect * 1.4)}
    frames_per_effect = strength_map.get(effect_strength, frames_per_effect)

    all_frames = []

    st.info("Generazione video in corso...")
    progress = st.progress(0)

    frame_counter = 0
    total_expected_frames = frames_per_effect * total_transitions

    # Mappa scelta effetto a funzione
    effect_map = {
        "Fade": fade_effect,
        "Morph": morph_effect,
        "Glitch": glitch_effect,
        "Pixel Block": pixel_block_effect,
        "VHS": vhs_effect,
        "Corrupted Lines": corrupted_lines_effect,
        "Random": random_effect
    }

    effect_func = effect_map.get(effect_choice, random_effect)

    for i in range(total_transitions):
        img1, img2 = images[i], images[i+1]
        frames = effect_func(img1, img2, frames_per_effect)
        all_frames.extend(frames)

        frame_counter += len(frames)
        progress.progress(min(frame_counter / total_expected_frames, 1.0))

    # Salvataggio video temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    writer = imageio.get_writer(filepath, fps=fps, codec="libx264", quality=8)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    st.success(f"✅ Video generato: durata ~{video_duration}s, {len(all_frames)} frame totali.")
    st.download_button("⬇️ Scarica video", data=open(filepath, "rb").read(), file_name="frame_to_frame_video.mp4", mime="video/mp4")

    # Rimuovo preview per alleggerire
    # st.video(filepath)

    # Pulizia file temporaneo dopo uso
    try:
        os.remove(filepath)
    except Exception:
        pass

else:
    st.warning("Carica almeno due immagini per iniziare.")
