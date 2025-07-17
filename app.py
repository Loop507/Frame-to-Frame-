import streamlit as st
import numpy as np
import cv2
import os
import random
import imageio
from PIL import Image
from pathlib import Path
import tempfile

st.set_page_config(page_title="🎞️ Frame-to-Frame FX", layout="wide")
st.title("🎞️ Frame-to-Frame FX Generator")

# --- EFFECT FUNCTIONS ---
def fade_effect(img1, img2, num_frames):
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

def slide_effect(img1, img2, num_frames):
    frames = []
    height, width, _ = img1.shape
    for i in range(num_frames):
        dx = int(width * i / num_frames)
        frame = np.zeros_like(img1)
        frame[:, :width - dx] = img1[:, dx:]
        frame[:, width - dx:] = img2[:, :dx]
        frames.append(frame)
    return frames

def glitch_effect(img1, img2, num_frames):
    frames = []
    for i in range(num_frames):
        frame = img1.copy()
        for _ in range(10):
            y = random.randint(0, img1.shape[0] - 1)
            h = random.randint(1, 10)
            x_shift = random.randint(-20, 20)
            frame[y:y+h] = np.roll(frame[y:y+h], x_shift, axis=1)
        alpha = i / (num_frames - 1)
        frame = cv2.addWeighted(frame, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

def morph_effect(img1, img2, num_frames):
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

def random_effect(img1, img2, num_frames):
    effects = [fade_effect, slide_effect, glitch_effect, morph_effect]
    selected = random.choice(effects)
    return selected(img1, img2, num_frames)

EFFECTS = {
    "Fade": fade_effect,
    "Slide": slide_effect,
    "Glitch": glitch_effect,
    "Morph": morph_effect,
    "Random": random_effect
}

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("📸 Input Options")
uploaded_files = st.sidebar.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
effect_name = st.sidebar.selectbox("Transition Effect", list(EFFECTS.keys()))
num_frames = st.sidebar.slider("Frames per Transition", 5, 60, 20)
output_format = st.sidebar.selectbox("Output Format", ["mp4", "gif", "webm"])

# --- MAIN LOGIC ---
if uploaded_files and len(uploaded_files) >= 2:
    images = [cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR) for f in uploaded_files]
    all_frames = []

    preview_img = Image.open(uploaded_files[0]).resize((200, 200))
    st.sidebar.image(preview_img, caption="Preview", use_column_width=False)

    progress_bar = st.progress(0, text="Generating frames...")

    for i in range(len(images) - 1):
        img1 = cv2.resize(images[i], (512, 512))
        img2 = cv2.resize(images[i + 1], (512, 512))
        transition = EFFECTS[effect_name](img1, img2, num_frames)
        all_frames.extend(transition)
        progress_bar.progress((i + 1) / (len(images) - 1), text=f"Processed {i + 1}/{len(images) - 1} transitions")

    progress_bar.empty()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmpfile:
        filepath = tmpfile.name
        if output_format == "gif":
            imageio.mimsave(filepath, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_frames], fps=24)
        else:
            writer = imageio.get_writer(filepath, fps=24)
            for f in all_frames:
                writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            writer.close()

    with open(filepath, "rb") as f:
        st.download_button("⬇️ Download Video", f, file_name=f"output.{output_format}", mime=f"video/{output_format}")

else:
    st.info("📥 Upload at least two images to begin.")
