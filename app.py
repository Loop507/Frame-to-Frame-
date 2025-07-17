import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import random

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

def slide_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, c = img1.shape
    frames = []
    for i in range(num_frames):
        offset = int(w * i / num_frames)
        frame = np.zeros_like(img1)
        if offset < w:
            frame[:, :w - offset] = img1[:, offset:]
        if offset > 0:
            frame[:, w - offset:] = img2[:, :offset]
        frames.append(frame)
    return frames

def zoom_effect(img1, img2, num_frames):
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, c = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        scale1 = 1 - 0.5 * alpha
        scale2 = 0.5 + 0.5 * alpha

        resized1 = cv2.resize(img1, None, fx=scale1, fy=scale1)
        canvas1 = np.zeros_like(img1)
        y1 = (h - resized1.shape[0]) // 2
        x1 = (w - resized1.shape[1]) // 2
        canvas1[y1:y1+resized1.shape[0], x1:x1+resized1.shape[1]] = resized1

        resized2 = cv2.resize(img2, None, fx=scale2, fy=scale2)
        canvas2 = np.zeros_like(img2)
        y2 = (h - resized2.shape[0]) // 2
        x2 = (w - resized2.shape[1]) // 2
        canvas2[y2:y2+resized2.shape[0], x2:x2+resized2.shape[1]] = resized2

        frame = cv2.addWeighted(canvas1, 1 - alpha, canvas2, alpha, 0)
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_block_effect(img1, img2, num_frames):
    # Transizione pixel block random per ogni frame
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    block_size = 16
    for alpha in np.linspace(0, 1, num_frames):
        frame = img1.copy()
        blocks = int((h * w) / (block_size * block_size))
        for _ in range(int(blocks * alpha)):
            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            frame[y:y+block_size, x:x+block_size] = img2[y:y+block_size, x:x+block_size]
        frames.append(frame)
    return frames

def vhs_effect(img1, img2, num_frames):
    # VHS-style glitch + scanlines + noise
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # scanlines
        for y in range(0, h, 2):
            frame[y, :, :] = (frame[y, :, :] * 0.6).astype(np.uint8)
        # noise
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        # random horizontal shift flicker
        if random.random() > 0.8:
            shift = random.randint(-15, 15)
            y = random.randint(0, h - 5)
            frame[y:y+5, :, :] = np.roll(frame[y:y+5, :, :], shift, axis=1)
        frames.append(frame)
    return frames

def corrupt_lines_effect(img1, img2, num_frames):
    # Linee orizzontali pixelate / spostate casualmente
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        num_lines = int(h * 0.05)
        for _ in range(num_lines):
            y = random.randint(0, h - 1)
            offset = random.randint(-30, 30)
            line = frame[y:y+1, :, :].copy()
            line = np.roll(line, offset, axis=1)
            frame[y:y+1, :, :] = line
        frames.append(frame)
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [
        fade_effect,
        morph_effect,
        glitch_effect,
        slide_effect,
        zoom_effect,
        pixel_block_effect,
        vhs_effect,
        corrupt_lines_effect
    ]
    effect = random.choice(effect_list)
    return effect(img1, img2, num_frames)


# --- STREAMLIT UI ---
st.title("🎞️ Frame-to-Frame FX Video Generator")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Carica le immagini (min. 2)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    output_format = st.selectbox("Formato output", ["1:1", "9:16", "16:9"])

num_seconds = st.slider("Durata totale video (secondi)", 1, 30, 10)
effect_choice = st.selectbox("Effetto", [
    "Fade", "Morph", "Glitch", "Slide", "Zoom", "Pixel Block", "VHS", "Corrupt Lines", "Random"
])
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

    total_transitions = len(images) - 1
    fps = 24
    total_frames = fps * num_seconds

    intensity_map = {"Soft": 0.5, "Medio": 1.0, "Hard": 1.5}
    intensity_factor = intensity_map[effect_strength]

    base_frames_per_transition = max(int(total_frames // total_transitions), 1)
    frames_per_transition = int(base_frames_per_transition * intensity_factor)

    all_frames = []

    st.info("Generazione video in corso...")
    progress = st.progress(0)

    for i in range(total_transitions):
        img1, img2 = images[i], images[i+1]

        if effect_choice == "Fade":
            frames = fade_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Morph":
            frames = morph_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Glitch":
            frames = glitch_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Slide":
            frames = slide_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Zoom":
            frames = zoom_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Pixel Block":
            frames = pixel_block_effect(img1, img2, frames_per_transition)
        elif effect_choice == "VHS":
            frames = vhs_effect(img1, img2, frames_per_transition)
        elif effect_choice == "Corrupt Lines":
            frames = corrupt_lines_effect(img1, img2, frames_per_transition)
        else:
            frames = random_effect(img1, img2, frames_per_transition)

        all_frames.extend(frames)
        progress.progress((i + 1) / total_transitions)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    height, width = target_size[1], target_size[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    for frame in all_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()

    st.success("✅ Video generato con successo!")

    # Preview molto piccola (15%)
    preview_img = all_frames[0]
    preview_pil = Image.fromarray(preview_img).resize((int(width * 0.15), int(height * 0.15)))
    st.image(preview_pil, caption="Preview frame iniziale", use_container_width=False)

    st.video(filepath)

else:
    st.warning("Carica almeno due immagini per iniziare.")
