import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Multi-Image", layout="wide")

# --- Effetti ---

def fade_effect(img1, img2, num_frames):
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
        frames.append(frame)
    return frames

def zoom_effect(img1, img2, num_frames, zoom_factor=1.2):
    frames = []
    h, w = img1.shape[:2]
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        scale = 1 + (zoom_factor - 1) * alpha
        new_h, new_w = int(h * scale), int(w * scale)
        img1_scaled = np.array(Image.fromarray(img1).resize((new_w, new_h)))
        img2_scaled = np.array(Image.fromarray(img2).resize((new_w, new_h)))
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        frame = (img1_scaled[offset_y:offset_y+h, offset_x:offset_x+w] * (1 - alpha) +
                 img2_scaled[offset_y:offset_y+h, offset_x:offset_x+w] * alpha).astype(np.uint8)
        frames.append(frame)
    return frames

def pixel_swap_random(img1, img2, num_frames):
    frames = []
    h, w, c = img1.shape
    total_pixels = h * w
    img1_copy = img1.copy()
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swapped = int(total_pixels * alpha)
        frame = img1_copy.copy()
        ys = np.random.randint(0, h, size=num_swapped)
        xs = np.random.randint(0, w, size=num_swapped)
        frame[ys, xs] = img2[ys, xs]
        frames.append(frame)
    return frames

def pixel_swap_blocks(img1, img2, num_frames, block_size=8):
    frames = []
    h, w, c = img1.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w
    swapped_blocks = set()
    img1_copy = img1.copy()
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_to_swap = int(total_blocks * alpha)
        frame = img1_copy.copy()
        while len(swapped_blocks) < num_to_swap:
            by = np.random.randint(0, blocks_h)
            bx = np.random.randint(0, blocks_w)
            swapped_blocks.add((by,bx))
        for by, bx in swapped_blocks:
            y_start = by * block_size
            y_end = y_start + block_size
            x_start = bx * block_size
            x_end = x_start + block_size
            frame[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
        frames.append(frame)
    return frames

def pixel_swap_wave(img1, img2, num_frames):
    frames = []
    h, w, c = img1.shape
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        wave_pos = int(w * alpha)
        frame = img1.copy()
        for y in range(h):
            wave_offset = int(10 * np.sin(y * 0.1 + i * 0.3))
            x_thr = wave_pos + wave_offset
            if x_thr > 0:
                end_x = min(x_thr, w)
                frame[y, :end_x] = img2[y, :end_x]
        frames.append(frame)
    return frames

def pixel_swap_spiral(img1, img2, num_frames):
    frames = []
    h, w, c = img1.shape
    center_y, center_x = h // 2, w // 2
    spiral_coords = []
    max_radius = int(np.sqrt(center_x**2 + center_y**2))
    for radius in range(max_radius + 1):
        steps = max(8, radius*2)
        for angle in np.linspace(0, 2*np.pi, steps, endpoint=False):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                spiral_coords.append((y, x))
    img1_copy = img1.copy()
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swapped = int(len(spiral_coords) * alpha)
        frame = img1_copy.copy()
        for j in range(num_swapped):
            y, x = spiral_coords[j]
            frame[y, x] = img2[y, x]
        frames.append(frame)
    return frames

# -- Funzione per creare video --
def create_video(frames, output_path, fps=30):
    if not frames:
        return False
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()
    return True

# --- UI ---
st.title("🎞️ Frame-to-Frame FX Multi-Image")

st.markdown("""
Carica da 2 a 5 immagini (jpg/png), scegli un effetto e genera un video MP4 in loop con transizioni.
""")

uploaded_files = st.file_uploader("📁 Carica immagini (2-5)", accept_multiple_files=True,
                                  type=["jpg", "jpeg", "png"])

effects_list = [
    "Dissolvenza (fade)",
    "Zoom",
    "Pixel casuali",
    "Blocchi",
    "Onda",
    "Spirale"
]

effect = st.selectbox("🎨 Seleziona effetto", effects_list)
fps = st.slider("🎚 FPS (frame per secondo)", 5, 60, 30)
frames_per_transition = st.slider("⏳ Frame per transizione", 10, 200, 60)

generate = st.button("🎬 Genera Video")

if generate:
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("Carica almeno 2 immagini per generare il video!")
    else:
        with st.spinner("Elaborazione video..."):
            size = (640, 480)
            imgs = []
            try:
                for f in uploaded_files[:5]:  # max 5 immagini
                    img = Image.open(f).convert('RGB').resize(size)
                    imgs.append(np.array(img))
            except Exception as e:
                st.error(f"Errore nel caricamento immagini: {e}")
                st.stop()

            all_frames = []
            n_imgs = len(imgs)

            effect_func = {
                "Dissolvenza (fade)": fade_effect,
                "Zoom": zoom_effect,
                "Pixel casuali": pixel_swap_random,
                "Blocchi": pixel_swap_blocks,
                "Onda": pixel_swap_wave,
                "Spirale": pixel_swap_spiral
            }[effect]

            # Genera transizioni a loop tra tutte le immagini caricate
            for i in range(n_imgs):
                img1 = imgs[i]
                img2 = imgs[(i + 1) % n_imgs]
                transition_frames = effect_func(img1, img2, frames_per_transition)
                all_frames.extend(transition_frames)

            # Salva video in cartella temporanea
            output_path = "frame_to_frame_output.mp4"
            success = create_video(all_frames, output_path, fps)

            if success:
                st.success("✅ Video generato!")
                with open(output_path, "rb") as f:
                    st.download_button("📥 Scarica Video MP4", f, file_name="frame_to_frame.mp4", mime="video/mp4")
            else:
                st.error("❌ Errore durante la generazione del video.")
