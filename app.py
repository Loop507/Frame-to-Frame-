import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import tempfile
import random

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Video Generator by Loop507", layout="wide")

# --- Effetti base con livelli di intensità ---

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
    max_shift = intensity
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(int(max_shift)):
            y = random.randint(0, h - 2)
            shift = random.randint(-max_shift, max_shift)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], shift, axis=1)
        frames.append(frame.astype(np.uint8))
    return frames

def pixelate_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    scale_min = max(2, 10 - intensity//5)
    scale_max = max(3, 15 - intensity//3)
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        scale = random.randint(scale_min, scale_max)
        small = cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(frame)
    return frames

def corrupt_lines_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    max_width = min(15, max(5, intensity // 3))
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for _ in range(intensity):
            x = random.randint(0, w - max_width - 1)
            y = random.randint(0, h - 1)
            width = random.randint(5, max_width)
            frame[y:y+1, x:x+width] = np.random.randint(0, 255, (1, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames

def wave_distort_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    amp = intensity * 2
    freq = 2 * np.pi / 50
    for alpha in np.linspace(0, 1, num_frames):
        blend = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        distorted = np.zeros_like(blend)
        for y in range(h):
            offset = int(amp * np.sin(freq * y))
            distorted[y] = np.roll(blend[y], offset, axis=0)
        frames.append(distorted)
    return frames

def zoom_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    max_zoom = 1 + intensity / 50.0
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        zoom = 1 + (max_zoom - 1) * (i / (num_frames - 1))
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
    zoom_min = 1
    zoom_max = 1 + intensity / 40.0
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        zoom = random.uniform(zoom_min, zoom_max)
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        zimg1 = cv2.warpAffine(img1, M, (w, h))
        zimg2 = cv2.warpAffine(img2, M, (w, h))
        frame = cv2.addWeighted(zimg1, 1 - alpha, zimg2, alpha, 0)
        frames.append(frame.astype(np.uint8))
    return frames

def color_echo_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        r = frame.copy()
        g = frame.copy()
        b = frame.copy()
        r[:, :, 1:] = 0
        g[:, :, [0, 2]] = 0
        b[:, :, :2] = 0
        merged = cv2.addWeighted(r, 0.5, g, 0.5, 0)
        merged = cv2.addWeighted(merged, 0.5, b, 0.5, 0)
        frames.append(merged.astype(np.uint8))
    return frames

def particle_float_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    particles = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(50)]
    for alpha in np.linspace(0, 1, num_frames):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        for i, (x, y) in enumerate(particles):
            dx, dy = random.randint(-intensity, intensity), random.randint(-intensity, intensity)
            nx = min(w - 1, max(0, x + dx))
            ny = min(h - 1, max(0, y + dy))
            cv2.circle(frame, (nx, ny), 2, (255, 255, 255), -1)
            particles[i] = (nx, ny)
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

def cinematic_morph_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    def ease(t): return t*t*(3 - 2*t)
    for i in range(num_frames):
        t = ease(i / (num_frames - 1))
        zoom = 1 + 0.1 * (1 - abs(0.5 - t) * 2)
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        zimg1 = cv2.warpAffine(img1, M, (w, h))
        zimg2 = cv2.warpAffine(img2, M, (w, h))
        frame = cv2.addWeighted(zimg1, 1 - t, zimg2, t, 0)
        frames.append(frame)
    return frames

def liquid_warp_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        flow = np.random.randn(h, w, 2).astype(np.float32) * intensity
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow[:, :, 0]).astype(np.float32)
        map_y = (map_y + flow[:, :, 1]).astype(np.float32)
        warped = cv2.remap(img1, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        frame = cv2.addWeighted(warped, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

def grid_dissolve_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    block_size = 32
    frames = []
    mask = np.zeros((h, w), np.uint8)
    for frame_id in range(num_frames):
        mask.fill(0)
        threshold = frame_id / num_frames
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if random.random() < threshold:
                    mask[y:y+block_size, x:x+block_size] = 255
        mask3 = cv2.merge([mask]*3)
        frame = np.where(mask3 == 255, img2, img1)
        frames.append(frame)
    return frames

def light_streaks_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    kernel = np.ones((1, intensity), np.float32) / intensity
    for alpha in np.linspace(0, 1, num_frames):
        blend = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        streaked = cv2.filter2D(blend, -1, kernel)
        frames.append(streaked)
    return frames

def motion_blur_effect(img1, img2, num_frames, intensity):
    h, w, _ = img1.shape
    frames = []
    max_shift = intensity
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        dx = int(max_shift * alpha)
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        shifted = cv2.warpAffine(img1, M, (w, h))
        frame = cv2.addWeighted(shifted, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

# --- Multi effetto combinato con livello (soft/medium/hard) ---

def combined_effect(img1, img2, num_frames, intensity, level):
    # Definisce effetti da combinare per livello
    level_map = {
        "Soft": [fade_effect, morph_effect],
        "Medium": [glitch_effect, pixelate_effect, wave_distort_effect],
        "Hard": [corrupt_lines_effect, particle_float_effect, distorted_slide_effect, liquid_warp_effect]
    }
    chosen_effects = level_map.get(level, [fade_effect])

    # Distribuisce i frames equamente tra effetti
    per_effect_frames = max(1, num_frames // len(chosen_effects))
    all_frames = []
    for i, effect in enumerate(chosen_effects):
        frames = effect(img1, img2, per_effect_frames, intensity)
        all_frames.extend(frames)
    # Se avanzano frame, li aggiunge con fade
    remaining = num_frames - len(all_frames)
    if remaining > 0:
        all_frames.extend(fade_effect(img1, img2, remaining, intensity))
    return all_frames

# --- Effetti disponibili e livelli ---

effects_simple = {
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
    "Distorted Slide": distorted_slide_effect,
    "Cinematic Morph": cinematic_morph_effect,
    "Liquid Warp": liquid_warp_effect,
    "Grid Dissolve": grid_dissolve_effect,
    "Light Streaks": light_streaks_effect,
    "Motion Blur": motion_blur_effect
}

levels = ["Soft", "Medium", "Hard"]

# --- UI ---

st.title("🎞️ Frame-to-Frame FX Video Generator by Loop507")

uploaded_files = st.file_uploader("Carica almeno 2 immagini", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

effect_mode = st.radio("Modalità effetti:", ["Singolo effetto", "Multi-effetto combinato"])

if effect_mode == "Singolo effetto":
    effect_choice = st.selectbox("Scegli un effetto", list(effects_simple.keys()))
else:
    level_choice = st.selectbox("Scegli il livello di intensità (combinato)", levels)

format_choice = st.selectbox("Formato video", ["1:1", "16:9", "9:16"])

duration = st.slider("Durata video (secondi)", 1, 20, 5)

intensity = st.slider("Intensità effetto", 1, 50, 15)

generate_btn = st.button("🎬 Genera Video")

if generate_btn:
    if not uploaded_files or len(uploaded_files) < 2:
        st.warning("Carica almeno 2 immagini!")
    else:
        format_dims = {
            "1:1": (512, 512),
            "16:9": (640, 360),
            "9:16": (360, 640)
        }
        target_size = format_dims[format_choice]

        images = [np.array(Image.open(file).convert("RGB")) for file in uploaded_files]
        images = [cv2.resize(img, target_size) for img in images]

        frames_per_transition = int(24 * duration / (len(images) - 1))
        all_frames = []
        progress = st.progress(0)

        for i in range(len(images) - 1):
            img1, img2 = images[i], images[i + 1]
            if effect_mode == "Singolo effetto":
                frames = effects_simple[effect_choice](img1, img2, frames_per_transition, intensity)
            else:
                frames = combined_effect(img1, img2, frames_per_transition, intensity, level_choice)
            all_frames.extend(frames)
            progress.progress(int((i + 1) / (len(images) - 1) * 100))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            filepath = tmpfile.name

        writer = imageio.get_writer(filepath, fps=24)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()

        with open(filepath, "rb") as f:
            st.download_button("📥 Scarica Video", f, file_name="output.mp4", mime="video/mp4")

        st.success("✅ Video generato con successo!")
else:
    st.info("Carica almeno due immagini e premi 'Genera Video'.")

