import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import tempfile
import random

# CONFIGURAZIONE ORIGINALE
st.set_page_config(page_title="Frame-to-Frame FX Video Generator by Loop507", layout="wide")

# --- MOTORE DISTORSIONI ---

def apply_scanlines(img, intensity, orientation="Orizzontale"):
    res = img.copy()
    gap = max(2, 12 - (int(intensity) // 8))
    if orientation == "Orizzontale":
        res[::gap, :, :] = res[::gap, :, :] // 2
    else:
        res[:, ::gap, :] = res[:, ::gap, :] // 2
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

# --- ALGORITMI DI TRANSIZIONE COMPLETI ---

def kinetic_conveyor_effect(all_images, num_frames, intensity, orientation, strand_width, speed_mult):
    h, w, _ = all_images[0].shape
    frames = []
    dim_total = h if orientation == "Orizzontale" else w
    num_strands = max(2, dim_total // strand_width)
    strand_size = dim_total // num_strands
    
    speeds = [random.uniform(0.01, 0.1) * speed_mult for _ in range(num_strands)]
    offsets = [random.uniform(0, len(all_images)) for _ in range(num_strands)]
    
    for f in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        noise_boost = 8.0 if random.random() > 0.96 else 1.0
        for s in range(num_strands):
            offsets[s] += speeds[s] * noise_boost
            idx = int(offsets[s] % len(all_images))
            nxt = (idx + 1) % len(all_images)
            alpha = offsets[s] % 1
            start, end = s*strand_size, (s+1)*strand_size if s < num_strands-1 else dim_total
            
            if orientation == "Orizzontale":
                strip = cv2.addWeighted(all_images[idx][start:end, :], 1-alpha, all_images[nxt][start:end, :], alpha, 0)
                frame[start:end, :] = strip
            else:
                strip = cv2.addWeighted(all_images[idx][:, start:end], 1-alpha, all_images[nxt][:, start:end], alpha, 0)
                frame[:, start:end] = strip
        if intensity > 40: frame = apply_scanlines(frame, intensity, orientation)
        frames.append(frame)
    return frames

def recursive_cut_effect(img1, img2, num_frames, intensity):
    frames = []
    curr = img1
    i = 0
    while i < num_frames:
        dur = random.randint(max(1, 10-(int(intensity)//10)), max(3, 20-(int(intensity)//10)))
        for _ in range(dur):
            if i < num_frames: frames.append(curr); i += 1
        curr = img2 if np.array_equal(curr, img1) else img1
    return frames

# --- INTERFACCIA UTENTE ---
st.title("🎞️ Frame-to-Frame FX Video Generator by Loop507")

uploaded_files = st.file_uploader("Carica Artefatti (Max 50)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# CONTROLLI LAYOUT STRUTTURATI
col1, col2, col3 = st.columns(3)

with col1:
    effect_choice = st.selectbox("Algoritmo", [
        "Kinetic Conveyor", "Recursive Cut", "Corrupted Lines", 
        "VHS Static", "Glitch Aberration", "Pixelate", "Classic Fade"
    ])
    format_choice = st.selectbox("Formato Video", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    orientation = st.radio("Direzione Strisce (solo Kinetic)", ["Orizzontale", "Verticale"])

with col2:
    duration = st.slider("Durata (secondi)", 1, 30, 10)
    strand_width = st.slider("Larghezza Strisce / Pixel", 2, 250, 50)
    speed_mult = st.slider("Velocità (solo Kinetic)", 0.5, 5.0, 1.0)

with col3:
    intensity = st.slider("Intensità FX", 1, 100, 35)
    st.info("Nota: 'Kinetic' usa tutte le foto, gli altri vanno in sequenza A->B.")

if st.button("🎬 GENERA VIDEO") and uploaded_files and len(uploaded_files) >= 2:
    if len(uploaded_files) > 50: uploaded_files = uploaded_files[:50]
    
    # Gestione formati
    format_map = {"16:9 (Orizzontale)": (1280, 720), "9:16 (Verticale)": (720, 1280), "1:1 (Quadrato)": (720, 720)}
    target_size = format_map[format_choice]

    with st.spinner("Processing immagini..."):
        images = [np.array(Image.open(f).convert("RGB").resize(target_size, Image.Resampling.NEAREST)) for f in uploaded_files]

    total_frames = 24 * duration
    all_frames = []

    if effect_choice == "Kinetic Conveyor":
        all_frames = kinetic_conveyor_effect(images, total_frames, intensity, orientation, strand_width, speed_mult)
    else:
        frames_per_trans = total_frames // (len(images) - 1)
        for i in range(len(images) - 1):
            i1, i2 = images[i], images[i+1]
            if effect_choice == "Recursive Cut":
                res = recursive_cut_effect(i1, i2, frames_per_trans, intensity)
            elif effect_choice == "Corrupted Lines":
                res = [apply_corruption(cv2.addWeighted(i1, 1-a, i2, a, 0), intensity) for a in np.linspace(0, 1, frames_per_trans)]
            elif effect_choice == "VHS Static":
                res = [cv2.add(cv2.addWeighted(i1, 1-a, i2, a, 0), np.random.randint(0, intensity, i1.shape, dtype='uint8')) for a in np.linspace(0, 1, frames_per_trans)]
            elif effect_choice == "Glitch Aberration":
                res = [apply_aberration(cv2.addWeighted(i1, 1-a, i2, a, 0), intensity) for a in np.linspace(0, 1, frames_per_trans)]
            elif effect_choice == "Pixelate":
                res = []
                for a in np.linspace(0, 1, frames_per_trans):
                    f = cv2.addWeighted(i1, 1-a, i2, a, 0)
                    s = max(2, strand_width // 4)
                    small = cv2.resize(f, (target_size[0]//s, target_size[1]//s), interpolation=cv2.INTER_LINEAR)
                    res.append(cv2.resize(small, target_size, interpolation=cv2.INTER_NEAREST))
            else: # Fade
                res = [cv2.addWeighted(i1, 1-a, i2, a, 0) for a in np.linspace(0, 1, frames_per_trans)]
            all_frames.extend(res)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        imageio.mimwrite(tmp.name, all_frames, fps=24, quality=6)
        st.video(tmp.name)
        with open(tmp.name, "rb") as f:
            st.download_button("💾 DOWNLOAD", f, file_name="loop507_artefact.mp4")
