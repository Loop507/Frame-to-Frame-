import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
import random
import traceback
import imageio

st.set_page_config(page_title="🎞️ Frame to Frame", layout="wide")

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
DEFAULT_FPS = 30
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "16:9 (Widescreen)": (16, 9),
    "9:16 (Portrait)": (9, 16),
    "4:3 (Classic)": (4, 3),
    "3:4 (Portrait Classic)": (3, 4),
    "21:9 (Ultrawide)": (21, 9),
    "Custom": None
}

EFFECT_INTENSITIES = {
    "Soft": 0.3,
    "Medium": 0.6,
    "Hard": 1.0
}

# --- Effetti (riutilizza gli stessi definiti nel messaggio precedente) ---
# Copia da linear_morph a pixel_morph (tutti gli effetti restano identici)
# e incollali qui, poi continua con il codice sotto:

EFFECTS_MAP = {
    "Linear": linear_morph,
    "Wave": wave_morph,
    "Spiral": spiral_morph,
    "Zoom": zoom_morph,
    "Glitch": glitch_morph,
    "Swap": swap_morph,
    "Pixelate": pixel_morph,
}

EFFECTS_WITH_RANDOM = list(EFFECTS_MAP.keys())

def load_image(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"Errore caricamento immagine {uploaded_file.name}: {str(e)}")
        return None

def calculate_size_from_ratio(ratio, base_size=800):
    w_ratio, h_ratio = ratio
    if w_ratio >= h_ratio:
        width = base_size
        height = int(base_size * h_ratio / w_ratio)
    else:
        height = base_size
        width = int(base_size * w_ratio / h_ratio)
    return width, height

def resize_to_target(img, target_size):
    try:
        original = Image.fromarray(img)
        target_ratio = target_size[0] / target_size[1]
        original_ratio = original.width / original.height

        if original_ratio > target_ratio:
            new_width = int(original.height * target_ratio)
            left = (original.width - new_width) // 2
            cropped = original.crop((left, 0, left + new_width, original.height))
        else:
            new_height = int(original.width / target_ratio)
            top = (original.height - new_height) // 2
            cropped = original.crop((0, top, original.width, top + new_height))

        return np.array(cropped.resize(target_size, Image.LANCZOS))
    except Exception as e:
        st.error(f"Errore nel ridimensionamento: {str(e)}")
        return None

def generate_morph(img1, img2, num_frames, effect, intensity):
    if effect == "Random":
        chosen_effect = random.choice(EFFECTS_WITH_RANDOM)
        morph_func = EFFECTS_MAP[chosen_effect]
    else:
        morph_func = EFFECTS_MAP.get(effect, linear_morph)
    try:
        return morph_func(img1, img2, num_frames, intensity)
    except Exception as e:
        st.error(f"Errore nel morphing con effetto {effect}: {str(e)}")
        st.error(traceback.format_exc())
        return []

def create_video(frames, fps, output_path, progress_callback=None):
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for i, frame in enumerate(frames):
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if progress_callback:
            progress_callback(i + 1, len(frames))
    video.release()
    return True

def create_preview_gif(frames, gif_path):
    gif_frames = frames[:min(5, len(frames))]
    imageio.mimsave(gif_path, gif_frames, fps=5)

def main():
    st.sidebar.title("Impostazioni")

    uploaded_files = st.sidebar.file_uploader("Carica almeno 2 immagini", type=SUPPORTED_FORMATS, accept_multiple_files=True)
    fps = st.sidebar.slider("FPS video", 1, 60, DEFAULT_FPS)
    aspect_ratio_choice = st.sidebar.selectbox("Rapporto d'aspetto video", list(ASPECT_RATIOS.keys()), index=1)

    custom_width, custom_height = None, None
    if aspect_ratio_choice == "Custom":
        custom_width = st.sidebar.number_input("Larghezza video", min_value=100, max_value=4000, value=800)
        custom_height = st.sidebar.number_input("Altezza video", min_value=100, max_value=4000, value=600)

    frames_per_transition = st.sidebar.slider("Frame per transizione", 1, 100, 30)
    effect_options = ["Random"] + list(EFFECTS_MAP.keys())
    effect = st.sidebar.selectbox("Effetto morphing", effect_options)
    intensity_label = st.sidebar.selectbox("Intensità effetto", list(EFFECT_INTENSITIES.keys()))
    intensity = EFFECT_INTENSITIES[intensity_label]

    if not uploaded_files or len(uploaded_files) < 2:
        st.warning("Carica almeno due immagini per procedere.")
        return

    images = []
    for f in uploaded_files:
        img = load_image(f)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        st.error("Non ci sono abbastanza immagini valide.")
        return

    if aspect_ratio_choice == "Custom" and custom_width and custom_height:
        target_size = (custom_width, custom_height)
    else:
        ratio = ASPECT_RATIOS.get(aspect_ratio_choice, (16, 9))
        target_size = calculate_size_from_ratio(ratio)

    for i in range(len(images)):
        resized = resize_to_target(images[i], target_size)
        if resized is None:
            st.error(f"Errore nel ridimensionamento immagine {i+1}")
            return
        images[i] = resized

    all_frames = []
    progress_bar = st.progress(0, text="Generazione morphing...")

    for i in range(len(images) - 1):
        frames = generate_morph(images[i], images[i+1], frames_per_transition, effect, intensity)
        if not frames:
            st.error(f"Errore nella transizione {i+1}")
            return
        all_frames.extend(frames)
        progress_bar.progress(int((i + 1) / (len(images) - 1) * 100), text="Generazione morphing...")

    st.success("Transizioni completate. Generazione video in corso...")

    with tempfile.TemporaryDirectory() as tmpdirname:
        video_path = os.path.join(tmpdirname, "output.mp4")
        gif_path = os.path.join(tmpdirname, "preview.gif")

        def progress_callback(done, total):
            progress_bar.progress(int(done / total * 100), text="Creazione video...")

        if create_video(all_frames, fps, video_path, progress_callback):
            create_preview_gif(all_frames, gif_path)
            st.image(gif_path, caption="Anteprima", use_column_width=True)
            st.download_button("Scarica il video", video_path, file_name="morph_video.mp4", mime="video/mp4")
        else:
            st.error("Errore durante la creazione del video.")

if __name__ == "__main__":
    main()
