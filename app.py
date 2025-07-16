import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import time
import random

st.set_page_config(page_title="🎞 Transizione Video", layout="centered")
st.title("🌀 Frame-to-Frame: Generatore Video da Immagini")

# === Funzioni effetti ===

def fade(img1, img2, steps):
    return [(img1 * (1 - a) + img2 * a).astype(np.uint8) for a in np.linspace(0, 1, steps)]

def zoom(img1, img2, steps, zoom_factor=1.2):
    h, w = img1.shape[:2]
    frames = []
    for i in range(steps):
        a = i / steps
        scale = 1 + a * (zoom_factor - 1)
        nh, nw = int(h * scale), int(w * scale)
        img1r = cv2.resize(img1, (nw, nh))
        img2r = cv2.resize(img2, (nw, nh))
        y, x = (nh - h) // 2, (nw - w) // 2
        crop1 = img1r[y:y+h, x:x+w]
        crop2 = img2r[y:y+h, x:x+w]
        blended = (crop1 * (1 - a) + crop2 * a).astype(np.uint8)
        frames.append(blended)
    return frames

def pixel_random(img1, img2, steps):
    h, w, _ = img1.shape
    total = h * w
    coords = [(y, x) for y in range(h) for x in range(w)]
    random.shuffle(coords)
    frames = []
    for i in range(steps):
        f = img1.copy()
        amount = int((i / steps) * total)
        for y, x in coords[:amount]:
            f[y, x] = img2[y, x]
        frames.append(f)
    return frames

def wave(img1, img2, steps):
    h, w, _ = img1.shape
    frames = []
    for i in range(steps):
        a = i / steps
        wave_pos = int(w * a)
        frame = img1.copy()
        for y in range(h):
            offset = int(10 * np.sin(y * 0.1 + i * 0.2))
            pos = wave_pos + offset
            if pos < w:
                frame[y, :pos] = img2[y, :pos]
        frames.append(frame)
    return frames

def spiral(img1, img2, steps):
    h, w, _ = img1.shape
    cx, cy = w // 2, h // 2
    coords = []
    for r in range(1, max(cx, cy)):
        for t in np.linspace(0, 2 * np.pi, r * 8):
            x = int(cx + r * np.cos(t))
            y = int(cy + r * np.sin(t))
            if 0 <= x < w and 0 <= y < h:
                coords.append((y, x))
    total = len(coords)
    frames = []
    for i in range(steps):
        f = img1.copy()
        count = int((i / steps) * total)
        for y, x in coords[:count]:
            f[y, x] = img2[y, x]
        frames.append(f)
    return frames

# === Utility ===

def save_video(frames, path, fps):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

def resize_image(img, size=(640, 480)):
    return np.array(Image.open(img).convert("RGB").resize(size))

# === UI ===

files = st.file_uploader("📁 Carica almeno 2 immagini", type=["jpg", "png"], accept_multiple_files=True)
fps = st.slider("🎞 FPS", 10, 60, 30)
steps = st.slider("⏱ Frame per transizione", 10, 60, 25)
effetto = st.selectbox("✨ Effetto di transizione", ["fade", "zoom", "pixel_random", "wave", "spiral"])
looping = st.checkbox("🔁 Loop finale (ultima immagine torna alla prima)")

if st.button("🚀 Genera Video") and files and len(files) >= 2:
    size = (640, 480)
    images = [resize_image(f, size) for f in files]
    all_frames = []
    progress_bar = st.progress(0)
    total_steps = len(images) - 1 + (1 if looping else 0)
    percent = 0

    for i in range(len(images) - 1):
        if effetto == "fade":
            frames = fade(images[i], images[i+1], steps)
        elif effetto == "zoom":
            frames = zoom(images[i], images[i+1], steps)
        elif effetto == "pixel_random":
            frames = pixel_random(images[i], images[i+1], steps)
        elif effetto == "wave":
            frames = wave(images[i], images[i+1], steps)
        elif effetto == "spiral":
            frames = spiral(images[i], images[i+1], steps)
        else:
            frames = fade(images[i], images[i+1], steps)

        all_frames.extend(frames)
        percent += 1 / total_steps
        progress_bar.progress(min(100, int(percent * 100)))

    if looping:
        frames = fade(images[-1], images[0], steps)
        all_frames.extend(frames)

    temp_path = os.path.join(tempfile.gettempdir(), "video_out.mp4")
    save_video(all_frames, temp_path, fps)
    st.success("✅ Video generato con successo!")
    st.video(temp_path)
