import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
import random
import traceback

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

def linear_morph(img1, img2, num_frames, intensity=1.0):
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        smooth_alpha = alpha * intensity + (1 - intensity) * 0.5
        morphed = (1 - smooth_alpha) * img1 + smooth_alpha * img2
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def wave_morph(img1, img2, num_frames, intensity=1.0):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        wave_intensity = 20 * intensity * np.sin(alpha * np.pi)
        dx = wave_intensity * np.sin(2 * np.pi * y / h + alpha * 4 * np.pi)
        dy = wave_intensity * np.cos(2 * np.pi * x / w + alpha * 4 * np.pi)
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def spiral_morph(img1, img2, num_frames, intensity=1.0):
    h, w = img1.shape[:2]
    frames = []
    cx, cy = w // 2, h // 2
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx = x - cx
        dy = y - cy
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        spiral_factor = alpha * 2 * np.pi * intensity
        angle_new = angle + spiral_factor * (radius / max(w, h))
        x_new = cx + radius * np.cos(angle_new)
        y_new = cy + radius * np.sin(angle_new)
        x_new = np.clip(x_new, 0, w - 1)
        y_new = np.clip(y_new, 0, h - 1)
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def zoom_morph(img1, img2, num_frames, intensity=1.0):
    h, w = img1.shape[:2]
    frames = []
    cx, cy = w // 2, h // 2
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        zoom_factor = 1 + (0.5 * intensity) * np.sin(alpha * np.pi)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom_factor)
        img1_t = cv2.warpAffine(img1, M, (w, h))
        img2_t = cv2.warpAffine(img2, M, (w, h))
        morphed = (1 - alpha) * img1_t + alpha * img2_t
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def glitch_morph(img1, img2, num_frames, intensity=1.0):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        base = (1 - alpha) * img1 + alpha * img2
        glitch_int = 0.3 * intensity * np.sin(alpha * np.pi * 4)
        if abs(glitch_int) > 0.1:
            result = base.copy()
            shift = int(glitch_int * 20 * intensity)
            if shift > 0:
                result[:, shift:, 0] = base[:, :-shift, 0]
            else:
                result[:, :shift, 0] = base[:, -shift:, 0]
            if shift > 0:
                result[:, :-shift, 2] = base[:, shift:, 2]
            else:
                result[:, -shift:, 2] = base[:, :shift, 2]
            num_lines = int(random.randint(1, 5) * intensity)
            for _ in range(num_lines):
                y = random.randint(0, h - 1)
                thickness = random.randint(1, 3)
                end_y = min(y + thickness, h)
                line_shift = int(random.randint(-30, 30) * intensity)
                if line_shift > 0:
                    result[y:end_y, line_shift:] = base[y:end_y, :-line_shift]
                elif line_shift < 0:
                    result[y:end_y, :line_shift] = base[y:end_y, -line_shift:]
            frames.append(np.clip(result, 0, 255).astype(np.uint8))
        else:
            frames.append(np.clip(base, 0, 255).astype(np.uint8))
    return frames

def swap_morph(img1, img2, num_frames, intensity=1.0):
    h, w = img1.shape[:2]
    frames = []
    block_size = max(2, min(w, h) // int(8 * intensity + 2))
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        result = img1.copy()
        num_blocks = int(alpha * intensity * (h // block_size) * (w // block_size))
        blocks_swapped = 0
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if blocks_swapped >= num_blocks:
                    break
                if random.random() < alpha * intensity:
                    y_end = min(y + block_size, h)
                    x_end = min(x + block_size, w)
                    result[y:y_end, x:x_end] = img2[y:y_end, x:x_end]
                    blocks_swapped += 1
            if blocks_swapped >= num_blocks:
                break
        frames.append(result)
    return frames

def pixel_morph(img1, img2, num_frames, intensity=1.0):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        pixel_level = max(1, int(2 + 30 * intensity * np.sin(alpha * np.pi)))
        base = (1 - alpha) * img1 + alpha * img2
        temp_img = cv2.resize(base, (w // pixel_level, h // pixel_level), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(temp_img, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(np.clip(pixelated, 0, 255).astype(np.uint8))
    return frames

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

def generate_morph(img1, img2, num_frames, effect, intensity):
    if effect == "Random":
        # Scegli casualmente un effetto per questa transizione
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

def create_video(frames, fps, output_path):
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()
    return True

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
    for i in range(len(images) - 1):
        frames = generate_morph(images[i], images[i+1], frames_per_transition, effect, intensity)
        if not frames:
            st.error(f"Errore nella transizione {i+1}")
            return
        all_frames.extend(frames)

    st.success("Morphing completato, generazione video in corso...")

    with tempfile.TemporaryDirectory() as tmpdirname:
        video_path = os.path.join(tmpdirname, "output.mp4")
        if create_video(all_frames, fps, video_path):
            st.download_button("Scarica il video", video_path, file_name="morph_video.mp4", mime="video/mp4")
        else:
            st.error("Errore durante la creazione del video.")

if __name__ == "__main__":
    main()
