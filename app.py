# app.py - Multi-image transition app

import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
from moviepy.editor import ImageSequenceClip
import tempfile
import random

st.set_page_config(page_title="Multi Image Transitions", layout="wide")

# --- EFFECTS ---

def fade_effect(img1, img2, num_frames):
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
            for alpha in np.linspace(0, 1, num_frames)]

def pixel_wave(img1, img2, num_frames):
    h, w, _ = img1.shape
    frames = []
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        frame = img1.copy()
        wave_pos = int(w * alpha)
        for y in range(h):
            offset = int(10 * np.sin(y * 0.1 + i * 0.3))
            x_end = min(w, wave_pos + offset)
            if x_end > 0:
                frame[y, :x_end] = img2[y, :x_end]
        frames.append(frame)
    return frames

EFFECTS = {
    "Dissolvenza": fade_effect,
    "Pixel Wave": pixel_wave,
}

# --- UTILITY ---

def load_image(path, size):
    img = Image.open(path).convert("RGB")
    return np.array(img.resize(size)).astype(np.float32)

def generate_transitions(images, effect_name, num_frames, loop_back):
    frames = []
    effect_fn = EFFECTS[effect_name]
    for i in range(len(images) - 1):
        frames += effect_fn(images[i], images[i + 1], num_frames)
    if loop_back:
        frames += effect_fn(images[-1], images[0], num_frames)
    return frames

# --- UI ---

st.title("🎞️ Slideshow con Transizioni ed Effetti")

uploaded_images = st.file_uploader("Carica 2 o più immagini", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

effect_choice = st.selectbox("Scegli effetto", list(EFFECTS.keys()))
num_frames = st.slider("Fotogrammi per transizione", 10, 120, 30)
loop_back = st.checkbox("Loop finale (ultima → prima immagine)")
generate = st.button("🎬 Genera video")

if generate and uploaded_images and len(uploaded_images) >= 2:
    with st.spinner("⏳ Elaborazione..."):

        size = (640, 480)
        images = [load_image(img, size) for img in uploaded_images]
        all_frames = generate_transitions(images, effect_choice, num_frames, loop_back)

        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "slideshow.mp4")
        clip = ImageSequenceClip([frame.astype(np.uint8) for frame in all_frames], fps=30)
        clip.write_videofile(output_path, codec='libx264', audio=False, verbose=False, logger=None)

    st.success("✅ Video pronto!")
    st.video(output_path)
else:
    st.info("Carica almeno due immagini per iniziare.")

