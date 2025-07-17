import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import traceback
import math
import random

# === CONFIGURAZIONE INIZIALE ===

st.set_page_config(page_title="🎞️ Frame to Frame", layout="wide")

# === COSTANTI ===

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

# === FUNZIONI DI BASE ===

def load_image(uploaded_file) -> Optional[np.ndarray]:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"Errore nel caricamento immagine: {str(e)}")
        return None

def calculate_size_from_ratio(ratio: Tuple[int, int], base_size: int = 800) -> Tuple[int, int]:
    w_ratio, h_ratio = ratio
    if w_ratio >= h_ratio:
        width = base_size
        height = int(base_size * h_ratio / w_ratio)
    else:
        height = base_size
        width = int(base_size * w_ratio / h_ratio)
    return width, height

def resize_to_target(img: np.ndarray, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    try:
        original = Image.fromarray(img)
        original_ratio = original.width / original.height
        target_ratio = target_size[0] / target_size[1]

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
        st.error(f"Errore ridimensionamento: {str(e)}")
        return None

# === EFFETTI DI MORPHING ===

# Qui puoi incollare tutte le funzioni dei morphing che già possiedi:
# es. linear_morph, wave_morph, spiral_morph, etc.
# (omessi per brevità, vedi tuo file precedente)

# === FUNZIONE DI GENERAZIONE CON PROGRESS ===

def generate_morph_with_progress(img1: np.ndarray, img2: np.ndarray, num_frames: int, effect: str, intensity: float = 1.0):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        effect_functions = {
            "Linear": linear_morph,
            "Wave": wave_morph,
            "Spiral": spiral_morph,
            "Zoom": zoom_morph,
            "Glitch": glitch_morph,
            "Swap": swap_morph,
            "Pixel": pixel_morph,
            "Distorted Lines": distorted_lines_morph,
            "Slice": slice_morph,
            "Rotation": rotation_morph,
            "Ripple": ripple_morph,
            "Random": random_morph
        }

        morph_function = effect_functions.get(effect, linear_morph)
        frames = morph_function(img1, img2, num_frames, intensity)

        for i in range(num_frames):
            progress = (i + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"Generazione frame {i+1}/{num_frames}")

        progress_bar.empty()
        status_text.empty()
        return frames
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Errore durante il morphing: {str(e)}")
        return []

# === SALVATAGGIO FILE ===

def save_as_gif(frames: List[np.ndarray], path: str, fps: int) -> bool:
    try:
        pil_frames = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        return True
    except Exception as e:
        st.error(f"Errore salvataggio GIF: {str(e)}")
        return False

def save_as_mp4(frames: List[np.ndarray], path: str, fps: int) -> bool:
    try:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return True
    except Exception as e:
        st.error(f"Errore salvataggio MP4: {str(e)}")
        return False

# === INTERFACCIA PRINCIPALE ===

def main():
    st.title("🎞️ Frame to Frame - Morphing Creator by Loop507")
    
    st.sidebar.header("⚙️ Impostazioni")
    aspect_choice = st.sidebar.selectbox("Aspect Ratio", list(ASPECT_RATIOS.keys()))
    
    if aspect_choice == "Custom":
        target_width = st.sidebar.number_input("Larghezza", 100, 1920, 800)
        target_height = st.sidebar.number_input("Altezza", 100, 1080, 600)
    else:
        ratio = ASPECT_RATIOS[aspect_choice]
        base_size = st.sidebar.slider("Dimensione Base", 400, 1200, 800, 50)
        target_width, target_height = calculate_size_from_ratio(ratio, base_size)
    
    duration = st.sidebar.slider("Durata (sec)", 1, 30, 5)
    fps = st.sidebar.slider("FPS", 10, 60, DEFAULT_FPS)
    num_frames = int(duration * fps)

    effect = st.sidebar.selectbox("Effetto", [
        "Linear", "Wave", "Spiral", "Zoom",
        "Glitch", "Swap", "Pixel", "Distorted Lines",
        "Slice", "Rotation", "Ripple", "Random"
    ])

    intensity_level = st.sidebar.selectbox("Intensità", list(EFFECT_INTENSITIES.keys()))
    intensity = EFFECT_INTENSITIES[intensity_level]
    
    export_format = st.sidebar.selectbox("Formato Output", ["MP4", "GIF"])

    uploaded_files = st.file_uploader("Carica almeno 2 immagini (max 10)", type=SUPPORTED_FORMATS, accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("Carica almeno due immagini")
            return
        if len(uploaded_files) > 10:
            st.error("Massimo 10 immagini")
            return

        images = []
        for file in uploaded_files:
            img = load_image(file)
            if img is not None:
                resized = resize_to_target(img, (target_width, target_height))
                if resized is not None:
                    images.append(resized)

        if len(images) < 2:
            st.error("Errore nel caricamento immagini")
            return

        if st.button("🚀 Genera Morphing"):
            transitions = len(images) - 1
            frames_per_transition = num_frames // transitions
            all_frames = []

            for i in range(transitions):
                st.subheader(f"Transizione {i+1}: immagine {i+1} → immagine {i+2}")
                frames = generate_morph_with_progress(
                    images[i], images[i+1],
                    frames_per_transition, effect, intensity
                )

                if not frames:
                    st.error(f"Errore nella transizione {i+1}")
                    return

                all_frames.extend(frames)

            if all_frames:
                st.success(f"{len(all_frames)} frame generati!")

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format.lower()}") as tmp_file:
                    success = save_as_mp4(all_frames, tmp_file.name, fps) if export_format == "MP4" else save_as_gif(all_frames, tmp_file.name, fps)

                    if success:
                        with open(tmp_file.name, "rb") as f:
                            st.download_button(
                                label=f"📥 Scarica {export_format}",
                                data=f.read(),
                                file_name=f"morph_{effect.lower()}.{export_format.lower()}",
                                mime="video/mp4" if export_format == "MP4" else "image/gif"
                            )
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass

if __name__ == "__main__":
    main()
