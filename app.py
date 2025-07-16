import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import random

st.set_page_config(page_title="🎥 Transizioni Video", layout="centered")

# === EFFETTI ===

def fade(img1, img2, frames):
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
            for alpha in np.linspace(0, 1, frames)]

def zoom(img1, img2, frames, zoom_factor=1.2):
    h, w = img1.shape[:2]
    output = []
    for i, alpha in enumerate(np.linspace(0, 1, frames)):
        scale = 1 + (zoom_factor - 1) * alpha
        nh, nw = int(h * scale), int(w * scale)
        img1_scaled = cv2.resize(img1, (nw, nh))
        img2_scaled = cv2.resize(img2, (nw, nh))
        ox, oy = (nw - w) // 2, (nh - h) // 2
        f = (img1_scaled[oy:oy+h, ox:ox+w] * (1 - alpha) +
             img2_scaled[oy:oy+h, ox:ox+w] * alpha).astype(np.uint8)
        output.append(f)
    return output

def pixel_random(img1, img2, frames):
    h, w, _ = img1.shape
    total = h * w
    frames_out = []
    for alpha in np.linspace(0, 1, frames):
        frame = img1.copy()
        idx = np.random.choice(total, int(total * alpha), replace=False)
        ys, xs = np.unravel_index(idx, (h, w))
        frame[ys, xs] = img2[ys, xs]
        frames_out.append(frame)
    return frames_out

def pixel_blocks(img1, img2, frames, block=16):
    h, w, _ = img1.shape
    blocks = [(y, x) for y in range(0, h, block) for x in range(0, w, block)]
    frames_out = []
    for alpha in np.linspace(0, 1, frames):
        n = int(len(blocks) * alpha)
        frame = img1.copy()
        for by, bx in random.sample(blocks, n):
            frame[by:by+block, bx:bx+block] = img2[by:by+block, bx:bx+block]
        frames_out.append(frame)
    return frames_out

def pixel_wave(img1, img2, frames):
    h, w, _ = img1.shape
    output = []
    for i, alpha in enumerate(np.linspace(0, 1, frames)):
        wave_pos = int(w * alpha)
        frame = img1.copy()
        for y in range(h):
            off = int(10 * np.sin(y * 0.1 + i * 0.3))
            x_end = min(w, wave_pos + off)
            if x_end > 0:
                frame[y, :x_end] = img2[y, :x_end]
        output.append(frame)
    return output

def spiral(img1, img2, frames):
    h, w, _ = img1.shape
    cx, cy = w // 2, h // 2
    coords = []
    rmax = int(np.sqrt(cx**2 + cy**2))
    for r in range(rmax):
        for theta in np.linspace(0, 2*np.pi, max(8, r*2), endpoint=False):
            x = int(cx + r * np.cos(theta))
            y = int(cy + r * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                coords.append((y, x))
    frames_out = []
    for alpha in np.linspace(0, 1, frames):
        n = int(len(coords) * alpha)
        frame = img1.copy()
        for y, x in coords[:n]:
            frame[y, x] = img2[y, x]
        frames_out.append(frame)
    return frames_out

# === UTILITY ===

def load_image(uploaded_file, size):
    img = Image.open(uploaded_file).convert("RGB").resize(size)
    return np.array(img).astype(np.uint8)

def save_video(frames, path, fps):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

# === STREAMLIT UI ===

st.title("🎬 Frame-to-Frame Slideshow")
imgs = st.file_uploader("📸 Carica almeno 2 immagini", type=["jpg", "png"], accept_multiple_files=True)
effect_name = st.selectbox("✨ Effetto", ["Dissolvenza", "Zoom", "Pixel Random", "Pixel Blocks", "Pixel Wave", "Spirale"])
fps = st.slider("🎞 FPS", 10, 60, 30)
frames_per = st.slider("📷 Fotogrammi per transizione", 10, 100, 30)
loop_back = st.checkbox("↩️ Loop finale")

if st.button("🎥 Genera Video") and imgs and len(imgs) >= 2:
    st.info("⏳ Elaborazione in corso...")
    size = (640, 480)
    images = [load_image(i, size) for i in imgs]
    effect_map = {
        "Dissolvenza": fade,
        "Zoom": zoom,
        "Pixel Random": pixel_random,
        "Pixel Blocks": pixel_blocks,
        "Pixel Wave": pixel_wave,
        "Spirale": spiral,
    }

    all_frames = []
    transitions = len(images) - 1 + (1 if loop_back else 0)
    progress = st.progress(0.0)

    for i in range(len(images) - 1):
        effect_fn = effect_map[effect_name]
        frames = effect_fn(images[i], images[i+1], frames_per)
        all_frames.extend(frames)
        progress.progress((i+1)/transitions)

    if loop_back:
        frames = effect_map[effect_name](images[-1], images[0], frames_per)
        all_frames.extend(frames)
        progress.progress(1.0)

    tmp_path = os.path.join(tempfile.gettempdir(), "video_finale.mp4")
    save_video(all_frames, tmp_path, fps)

    st.success("✅ Video creato!")
    st.video(tmp_path)
