import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import tempfile
import random

st.set_page_config(page_title="🎞️ RECURSIVE COLLAPSE: Ultimate Machine v3", layout="wide")

# --- STRUMENTI DI BASE (LAYER) ---

def apply_scanlines(img, intensity):
    res = img.copy()
    gap = max(2, 12 - (int(intensity) // 8))
    res[::gap, :, :] = res[::gap, :, :] // 2
    return res

def apply_aberration(img, intensity):
    shift = int(intensity // 6)
    b, g, r = cv2.split(img)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    return cv2.merge([b, g, r])

def apply_corruption(img, intensity):
    res = img.copy()
    h, w, _ = res.shape
    for _ in range(int(intensity // 2)):
        y = random.randint(0, h - 1)
        line_w = random.randint(1, 4)
        res[y:y+line_w, :] = np.roll(res[y:y+line_w, :], random.randint(-50, 50), axis=1)
    return res

# --- ALGORITMI DI TRANSIZIONE ---

def kinetic_conveyor_effect(all_images, num_frames, intensity):
    h, w, _ = all_images[0].shape
    frames = []
    num_strands = max(5, int(intensity))
    strand_size = h // num_strands
    speeds = [random.uniform(0.02, 0.15) for _ in range(num_strands)]
    offsets = [random.uniform(0, len(all_images)) for _ in range(num_strands)]
    
    for f in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        noise_boost = 12.0 if random.random() > 0.96 else 1.0
        for s in range(num_strands):
            offsets[s] += speeds[s] * noise_boost
            idx = int(offsets[s] % len(all_images))
            nxt = (idx + 1) % len(all_images)
            alpha = offsets[s] % 1
            y_s, y_e = s*strand_size, (s+1)*strand_size if s < num_strands-1 else h
            strip = cv2.addWeighted(all_images[idx][y_s:y_e, :], 1-alpha, all_images[nxt][y_s:y_e, :], alpha, 0)
            frame[y_s:y_e, :] = strip
        if intensity > 50: frame = apply_scanlines(frame, intensity)
        frames.append(frame)
    return frames

def recursive_cut_effect(img1, img2, num_frames, intensity):
    frames = []
    curr = img1
    i = 0
    while i < num_frames:
        dur = random.randint(max(1, 8-(int(intensity)//10)), max(3, 15-(int(intensity)//10)))
        for _ in range(dur):
            if i < num_frames: frames.append(curr); i += 1
        curr = img2 if np.array_equal(curr, img1) else img1
    return frames

def combi_analog_collapse(img1, img2, num_frames, intensity):
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        f = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
        f = apply_scanlines(f, intensity)
        f = apply_aberration(f, intensity)
        noise = np.random.randint(0, int(intensity), f.shape, dtype='uint8')
        frames.append(cv2.add(f, noise))
    return frames

def corrupted_lines_trans(img1, img2, num_frames, intensity):
    return [apply_corruption(cv2.addWeighted(img1, 1-a, img2, a, 0), intensity) for a in np.linspace(0, 1, num_frames)]

def pixelate_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        f = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
        s = max(2, 30 - (int(intensity) // 4))
        small = cv2.resize(f, (w//s, h//s), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))
    return frames

# --- CONFIGURAZIONE DIZIONARIO ---
effect_funcs = {
    "🎞️ Kinetic Conveyor (Tutte le foto)": kinetic_conveyor_effect,
    "✂️ Recursive Cut (A/B)": recursive_cut_effect,
    "🚀 COMBI: Analog Collapse": combi_analog_collapse,
    "📡 Corrupted Lines": corrupted_lines_trans,
    "📺 Scanlines Only": lambda i1, i2, n, f: [apply_scanlines(cv2.addWeighted(i1, 1-a, i2, a, 0), f) for a in np.linspace(0, 1, n)],
    "🌈 Glitch Aberration": lambda i1, i2, n, f: [apply_aberration(cv2.addWeighted(i1, 1-a, i2, a, 0), f) for a in np.linspace(0, 1, n)],
    "👾 Pixelate": pixelate_effect,
    "🌫️ Classic Fade": lambda i1, i2, n, f: [cv2.addWeighted(i1, 1-a, i2, a, 0).astype(np.uint8) for a in np.linspace(0, 1, n)]
}

# --- INTERFACCIA ---
st.title("🎞️ RECURSIVE COLLAPSE: Ultimate Machine")
st.write("Versione Revisionata: Massima stabilità per Streamlit.")

uploaded_files = st.file_uploader("Carica Artefatti (Max 50)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
effect_choice = st.selectbox("Algoritmo", list(effect_funcs.keys()))
format_choice = st.selectbox("Formato", ["16:9", "1:1", "9:16"])
duration = st.slider("Durata (secondi)", 1, 30, 10)
intensity = st.slider("Intensità", 1, 100, 35)

if st.button("🎬 GENERA") and uploaded_files and len(uploaded_files) >= 2:
    if len(uploaded_files) > 50: uploaded_files = uploaded_files[:50]
    
    dims = {"1:1": (720, 720), "16:9": (1280, 720), "9:16": (720, 1280)}
    target_size = dims[format_choice]

    with st.spinner("Processing immagini..."):
        images = [np.array(Image.open(f).convert("RGB").resize(target_size, Image.Resampling.NEAREST)) for f in uploaded_files]

    total_frames = 24 * duration
    all_frames = []

    if "Kinetic" in effect_choice:
        all_frames = kinetic_conveyor_effect(images, total_frames, intensity)
    else:
        frames_per_trans = total_frames // (len(images) - 1)
        for i in range(len(images) - 1):
            res = effect_funcs[effect_choice](images[i], images[i+1], frames_per_trans, intensity)
            all_frames.extend(res)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        imageio.mimwrite(tmp.name, all_frames, fps=24, quality=6)
        st.video(tmp.name)
        with open(tmp.name, "rb") as f:
            st.download_button("💾 DOWNLOAD", f, file_name="collapse_final.mp4")
