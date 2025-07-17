import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import random
import imageio

st.set_page_config(page_title="🎞️ Frame to Frame", layout="wide")

# Costanti
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
DEFAULT_FPS = 30
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "16:9 (Widescreen)": (16, 9),
    "9:16 (Portrait)": (9, 16),
    "Custom": None
}
EFFECT_INTENSITIES = {
    "Soft": 0.3,
    "Medium": 0.6,
    "Hard": 1.0
}

# --- Funzioni base ---

def load_image(uploaded_file) -> Optional[np.ndarray]:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"Errore caricamento immagine '{uploaded_file.name}': {e}")
        return None

def calculate_size_from_ratio(ratio: Tuple[int, int], base_size: int = 512) -> Tuple[int, int]:
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
        st.error(f"Errore ridimensionamento: {e}")
        return None

# --- Morph Effects ---

def linear_morph(img1, img2, num_frames, intensity):
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        smooth_alpha = alpha * intensity + (1 - intensity) * 0.5
        morphed = (1 - smooth_alpha) * img1 + smooth_alpha * img2
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def wave_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        wave_intensity = 20 * intensity * np.sin(alpha * np.pi)
        dx = wave_intensity * np.sin(2 * np.pi * y / h + alpha * 4 * np.pi)
        dy = wave_intensity * np.cos(2 * np.pi * x / w + alpha * 4 * np.pi)
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new, y_new, cv2.INTER_LINEAR)
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def spiral_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    center_x, center_y = w // 2, h // 2
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx = x - center_x
        dy = y - center_y
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        spiral_factor = alpha * 2 * np.pi * intensity
        angle_new = angle + spiral_factor * (radius / max(w, h))
        x_new = center_x + radius * np.cos(angle_new)
        y_new = center_y + radius * np.sin(angle_new)
        x_new = np.clip(x_new, 0, w - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, h - 1).astype(np.float32)
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new, y_new, cv2.INTER_LINEAR)
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def zoom_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    center_x, center_y = w // 2, h // 2
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        zoom_factor = 1 + (0.5 * intensity) * np.sin(alpha * np.pi)
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
        img1_t = cv2.warpAffine(img1, M, (w, h))
        img2_t = cv2.warpAffine(img2, M, (w, h))
        morphed = (1 - alpha) * img1_t + alpha * img2_t
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def glitch_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        base = (1 - alpha) * img1 + alpha * img2
        glitch_intensity = 0.3 * intensity * np.sin(alpha * np.pi * 4)
        if abs(glitch_intensity) > 0.1:
            result = base.copy()
            shift = int(glitch_intensity * 20 * intensity)
            if shift > 0:
                result[:, shift:, 0] = base[:, :-shift, 0]
                result[:, :-shift, 2] = base[:, shift:, 2]
            else:
                result[:, :shift, 0] = base[:, -shift:, 0]
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

def swap_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    block_size = min(w, h) // max(2, int(8 * intensity + 2))
    frames = []
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

def pixel_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        pixel_level = max(2, int(2 + 30 * intensity * np.sin(alpha * np.pi)))
        base = (1 - alpha) * img1 + alpha * img2
        small_h, small_w = h // pixel_level, w // pixel_level
        small_img = cv2.resize(base, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(np.clip(pixelated, 0, 255).astype(np.uint8))
    return frames

def distorted_lines_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        base = (1 - alpha) * img1 + alpha * img2
        result = base.copy()
        distortion_intensity = 0.5 * intensity * np.sin(alpha * np.pi * 3)
        if abs(distortion_intensity) > 0.1:
            line_spacing = max(1, int(10 / intensity))
            for y in range(0, h, line_spacing):
                shift = int(distortion_intensity * 20 * ((y / h) - 0.5))
                result[y:y+2, :] = np.roll(result[y:y+2, :], shift, axis=1)
        frames.append(np.clip(result, 0, 255).astype(np.uint8))
    return frames

def slice_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    slice_height = max(4, int(10 * intensity))
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        result = img1.copy()
        num_slices = int(alpha * 10 * intensity)
        for s in range(num_slices):
            y = (s * slice_height) % h
            if s % 2 == 0:
                y2 = min(y + slice_height, h)
                result[y:y2, :] = img2[y:y2, :]
        frames.append(result)
    return frames

def rotation_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    center = (w // 2, h // 2)
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        angle = alpha * 360 * intensity
        M1 = cv2.getRotationMatrix2D(center, angle, 1)
        M2 = cv2.getRotationMatrix2D(center, -angle, 1)
        img1_r = cv2.warpAffine(img1, M1, (w, h))
        img2_r = cv2.warpAffine(img2, M2, (w, h))
        morphed = (1 - alpha) * img1_r + alpha * img2_r
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def ripple_morph(img1, img2, num_frames, intensity):
    h, w = img1.shape[:2]
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ripple_amount = 5 * intensity * np.sin(alpha * 2 * np.pi)
        dx = ripple_amount * np.sin(2 * np.pi * y / 30)
        dy = ripple_amount * np.cos(2 * np.pi * x / 30)
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new, y_new, cv2.INTER_LINEAR)
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def random_morph(img1, img2, num_frames, intensity):
    effects = [
        linear_morph,
        wave_morph,
        spiral_morph,
        zoom_morph,
        glitch_morph,
        swap_morph,
        pixel_morph,
        distorted_lines_morph,
        slice_morph,
        rotation_morph,
        ripple_morph
    ]
    frames = []
    random_effect = random.choice(effects)
    frames.extend(random_effect(img1, img2, num_frames, intensity))
    return frames

# Mappa effetti per selezione
EFFECTS_MAP = {
    "Linear": linear_morph,
    "Wave": wave_morph,
    "Spiral": spiral_morph,
    "Zoom": zoom_morph,
    "Glitch": glitch_morph,
    "Swap": swap_morph,
    "Pixelate": pixel_morph,
    "Distorted Lines": distorted_lines_morph,
    "Slice": slice_morph,
    "Rotation": rotation_morph,
    "Ripple": ripple_morph,
    "Random": random_morph,
}

# --- Streamlit UI ---

st.title("🎞️ Frame-to-Frame Morphing Video Creator")

uploaded_imgs = st.file_uploader(
    "Carica almeno due immagini (jpg, png, bmp...)",
    type=SUPPORTED_FORMATS,
    accept_multiple_files=True
)

if uploaded_imgs and len(uploaded_imgs) >= 2:
    st.sidebar.header("Impostazioni video")

    # Seleziona effetto
    selected_effect = st.sidebar.selectbox(
        "Scegli effetto di morphing",
        list(EFFECTS_MAP.keys()),
        index=0
    )

    # Seleziona intensità
    selected_intensity_label = st.sidebar.selectbox(
        "Seleziona intensità effetto",
        list(EFFECT_INTENSITIES.keys()),
        index=1
    )
    intensity = EFFECT_INTENSITIES[selected_intensity_label]

    # Aspect ratio
    selected_ratio_label = st.sidebar.selectbox("Formato output (aspect ratio)", list(ASPECT_RATIOS.keys()))
    if selected_ratio_label == "Custom":
        custom_width = st.sidebar.number_input("Larghezza pixel", min_value=64, max_value=3840, value=512)
        custom_height = st.sidebar.number_input("Altezza pixel", min_value=64, max_value=3840, value=512)
        target_size = (custom_width, custom_height)
    else:
        ratio = ASPECT_RATIOS[selected_ratio_label]
        target_size = calculate_size_from_ratio(ratio, base_size=512)

    # Durata e FPS
    duration_sec = st.sidebar.slider("Durata del morphing (secondi)", min_value=1, max_value=20, value=5)
    fps = st.sidebar.slider("FPS (frame per secondo)", min_value=10, max_value=60, value=DEFAULT_FPS)

    # Output format
    output_format = st.sidebar.selectbox("Formato output", ["mp4", "gif"])

    if st.sidebar.button("Genera morphing"):
        # Carica e prepara immagini
        imgs = []
        for f in uploaded_imgs:
            img = load_image(f)
            if img is not None:
                resized = resize_to_target(img, target_size)
                if resized is not None:
                    imgs.append(resized)
        if len(imgs) < 2:
            st.error("Carica almeno due immagini valide!")
        else:
            # Genera frames
            num_frames = duration_sec * fps
            all_frames = []
            progress_bar = st.progress(0)
            total_steps = (len(imgs) - 1) * num_frames

            step = 0
            for i in range(len(imgs) - 1):
                func = EFFECTS_MAP[selected_effect]
                frames = func(imgs[i], imgs[i+1], num_frames, intensity)
                all_frames.extend(frames)
                # Aggiorna progress
                for _ in frames:
                    step += 1
                    progress_bar.progress(min(step / total_steps, 1.0))

            # Salva video/gif temporaneamente
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + output_format) as tmpfile:
                filepath = tmpfile.name

            try:
                if output_format == "mp4":
                    writer = imageio.get_writer(filepath, fps=fps, codec="libx264", quality=8)
                    for frame in all_frames:
                        writer.append_data(frame)
                    writer.close()
                elif output_format == "gif":
                    imageio.mimsave(filepath, all_frames, fps=fps)
                st.success(f"Video generato con successo! [Dimensione: {target_size[0]}x{target_size[1]} px]")

                # Preview miniatura (piccola, per non appesantire)
                preview_img = all_frames[0]
                st.image(preview_img, caption="Anteprima primo frame", use_container_width=True)

                # Download link
                with open(filepath, "rb") as file:
                    btn = st.download_button(
                        label="Scarica il video/gif",
                        data=file,
                        file_name=f"morphing.{output_format}",
                        mime=f"video/{output_format}" if output_format == "mp4" else "image/gif"
                    )
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

else:
    st.info("Carica almeno due immagini per abilitare il morphing.")

