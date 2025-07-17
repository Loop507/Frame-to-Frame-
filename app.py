import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
import random
from tqdm import tqdm

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Video Generator by Loop507", layout="wide")

# --- EFFETTI ---
def fade_effect(img1, img2, num_frames, intensity):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
            for alpha in np.linspace(0, 1, num_frames)]

def morph_effect(img1, img2, num_frames, intensity):
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8)
            for alpha in np.linspace(0, 1, num_frames)]

def glitch_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(intensity):
            y = random.randint(0, h - 2)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], random.randint(-intensity, intensity), axis=1)
        frames.append(frame.astype(np.uint8))
    return frames

def pixelate_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        scale = random.randint(4, 10)
        small = cv2.resize(frame, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(frame)
    return frames

def corrupt_lines_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(intensity):
            x = random.randint(0, w - 10)
            y = random.randint(0, h - 1)
            frame[y:y+1, x:x+random.randint(5, 15)] = np.random.randint(0, 255, (1, min(15, w - x), 3), dtype=np.uint8)
        frames.append(frame)
    return frames

def wave_distort_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        blend = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        distorted = np.zeros_like(blend)
        for y in range(h):
            offset = int(intensity * np.sin(2 * np.pi * y / 50))
            distorted[y] = np.roll(blend[y], offset, axis=0)
        frames.append(distorted)
    return frames

def zoom_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        zoom = 1 + (intensity / 100.0) * (i / num_frames)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 0, zoom)
        zoomed_img1 = cv2.warpAffine(img1, M, (w, h))
        zoomed_img2 = cv2.warpAffine(img2, M, (w, h))
        frame = cv2.addWeighted(zoomed_img1, 1 - alpha, zoomed_img2, alpha, 0)
        frames.append(frame.astype(np.uint8))
    return frames

def zoom_random_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        zoom = random.uniform(1, 1 + intensity / 50.0)
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        zimg1 = cv2.warpAffine(img1, M, (w, h))
        zimg2 = cv2.warpAffine(img2, M, (w, h))
        frame = cv2.addWeighted(zimg1, 1 - alpha, zimg2, alpha, 0)
        frames.append(frame)
    return frames

def color_echo_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        r = frame.copy(); g = frame.copy(); b = frame.copy()
        r[:, :, 1:] = 0; g[:, :, [0, 2]] = 0; b[:, :, :2] = 0
        merged = cv2.addWeighted(r, 0.5, g, 0.5, 0)
        merged = cv2.addWeighted(merged, 0.5, b, 0.5, 0)
        frames.append(merged.astype(np.uint8))
    return frames

def particle_float_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    particles = [(random.randint(0, w), random.randint(0, h)) for _ in range(50)]
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for x, y in particles:
            dx, dy = random.randint(-intensity, intensity), random.randint(-intensity, intensity)
            cv2.circle(frame, (min(w-1, max(0, x+dx)), min(h-1, max(0, y+dy))), 2, (255, 255, 255), -1)
        frames.append(frame)
    return frames

def distorted_slide_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for i in range(num_frames):
        alpha = i / num_frames
        offset = int((1 - alpha) * w)
        frame = np.zeros_like(img1)
        frame[:, :w - offset] = img2[:, offset:]
        frame[:, w - offset:] = img1[:, :offset]
        frames.append(frame)
    return frames

# --- EFFETTI DISPONIBILI ---
effect_funcs = {
    "Fade": fade_effect,
    "Morph": morph_effect,
    "Glitch": glitch_effect,
    "Pixelate": pixelate_effect,
    "Corrupt Lines": corrupt_lines_effect,
    "Wave Distort": wave_distort_effect,
    "Zoom": zoom_effect,
    "Zoom Random": zoom_random_effect,
    "Color Echo": color_echo_effect,
    "Particle Float": particle_float_effect,
    "Distorted Slide": distorted_slide_effect
}

# --- INTERFACCIA ---
st.title("🎞️ Frame-to-Frame FX Video Generator by Loop507")

uploaded_files = st.file_uploader("Carica almeno 2 immagini", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
effect_choice = st.selectbox("Scegli l'effetto", list(effect_funcs.keys()))
duration = st.slider("Durata video (in secondi)", 1, 20, 5)
intensity = st.slider("Intensità effetto", 1, 50, 15)
video_btn = st.button("🎬 Genera Video")

if video_btn and uploaded_files and len(uploaded_files) >= 2:
    images = [np.array(Image.open(file).convert("RGB")) for file in uploaded_files]
    images = [cv2.resize(img, (512, 512)) for img in images]

    frames_per_transition = int(24 * duration / (len(images) - 1))
    all_frames = []
    progress = st.progress(0.0)

    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        frames = effect_funcs[effect_choice](img1, img2, frames_per_transition, intensity)
        all_frames.extend(frames)
        progress.progress((i + 1) / (len(images) - 1))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        filepath = tmpfile.name

    writer = imageio.get_writer(filepath, fps=24)
    for frame in tqdm(all_frames):
        writer.append_data(frame)
    writer.close()

    with open(filepath, "rb") as f:
        st.download_button("📥 Scarica Video", f, file_name="output.mp4", mime="video/mp4")

    st.success("✅ Video generato con successo!")
else:
    st.info("Carica almeno due immagini e premi 'Genera Video'.")
