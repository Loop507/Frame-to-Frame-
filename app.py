import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# Configurazione pagina
st.set_page_config(page_title="Transizioni Video", layout="wide")

# --- Effetti disponibili ---

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
        frames.append(frame.astype(np.uint8))
    return frames

EFFECTS = {
    "Dissolvenza": fade_effect,
    "Pixel Wave": pixel_wave
}

# --- Utility ---

def load_image(file, size):
    img = Image.open(file).convert("RGB")
    return np.array(img.resize(size)).astype(np.float32)

def generate_transitions(images, effect_fn, num_frames, loop_back, progress_callback=None):
    frames = []
    total_transitions = len(images) - 1 + (1 if loop_back else 0)
    for i in range(len(images) - 1):
        if progress_callback:
            progress_callback(i / total_transitions)
        frames += effect_fn(images[i], images[i + 1], num_frames)
    if loop_back:
        if progress_callback:
            progress_callback(1.0)
        frames += effect_fn(images[-1], images[0], num_frames)
    return frames

def save_video(frames, path, fps=30):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

# --- Interfaccia Streamlit ---

st.title("🎥 Generatore di Video con Transizioni")
uploaded_images = st.file_uploader("Carica almeno 2 immagini", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images and len(uploaded_images) >= 2:
    effect = st.selectbox("Effetto di transizione", list(EFFECTS.keys()))
    frames_per_transition = st.slider("Fotogrammi per transizione", 10, 120, 30)
    loop = st.checkbox("Loop finale (ultima → prima)")
    fps = st.slider("FPS (velocità del video)", 10, 60, 30)
    generate_btn = st.button("🎬 Genera video")

    if generate_btn:
        with st.spinner("Creazione del video in corso..."):
            progress = st.progress(0.0)
            size = (640, 480)
            images = [load_image(img, size) for img in uploaded_images]
            frames = generate_transitions(
                images, EFFECTS[effect], frames_per_transition, loop,
                progress_callback=progress.progress
            )

            tmp_path = os.path.join(tempfile.gettempdir(), "video_output.mp4")
            save_video(frames, tmp_path, fps=fps)

        st.success("✅ Video generato!")
        st.video(tmp_path)
else:
    st.info("📸 Carica almeno due immagini per iniziare.")
