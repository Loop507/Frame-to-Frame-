import streamlit as st
import numpy as np
from PIL import Image
import cv2
import imageio
import random
import tempfile
import os

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Multi-Image", layout="wide")

# --- EFFETTI ---

def fade_effect(img1, img2, num_frames, intensity):
    # L’intensità regola la velocità di transizione, con più frame per soft e meno per hard
    factor = {'Soft': 1, 'Medium': 0.7, 'Hard': 0.4}[intensity]
    frames = int(num_frames * factor)
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8) for alpha in np.linspace(0, 1, frames)]

def morph_effect(img1, img2, num_frames, intensity):
    factor = {'Soft': 1, 'Medium': 0.7, 'Hard': 0.4}[intensity]
    frames = int(num_frames * factor)
    return [(cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)).astype(np.uint8) for alpha in np.linspace(0, 1, frames)]

def glitch_effect(img1, img2, num_frames, intensity):
    factor = {'Soft': 0.03, 'Medium': 0.06, 'Hard': 0.1}[intensity]
    frames = int(num_frames * (1-factor))
    h, w, _ = img1.shape
    glitch_frames = []
    for i in range(frames):
        frame = img1.copy()
        num_glitches = int(w * factor * 10)
        for _ in range(num_glitches):
            x = random.randint(0, w-20)
            y = random.randint(0, h-10)
            glitch_width = random.randint(10, 20)
            glitch_height = random.randint(5, 15)
            frame[y:y+glitch_height, x:x+glitch_width] = img2[y:y+glitch_height, x:x+glitch_width]
        glitch_frames.append(frame)
    return glitch_frames

def pixel_shuffle_effect(img1, img2, num_frames, intensity):
    factor = {'Soft': 3, 'Medium': 6, 'Hard': 10}[intensity]
    frames = int(num_frames * 0.8)
    h, w, c = img1.shape
    shuffled_frames = []
    for i in range(frames):
        frame = img1.copy()
        for _ in range(factor):
            x1, y1 = random.randint(0, w-10), random.randint(0, h-10)
            x2, y2 = random.randint(0, w-10), random.randint(0, h-10)
            block_w, block_h = random.randint(5, 15), random.randint(5, 15)
            temp = frame[y1:y1+block_h, x1:x1+block_w].copy()
            frame[y1:y1+block_h, x1:x1+block_w] = frame[y2:y2+block_h, x2:x2+block_w]
            frame[y2:y2+block_h, x2:x2+block_w] = temp
        shuffled_frames.append(frame)
    return shuffled_frames

def random_effect(img1, img2, num_frames, intensity):
    effects = [fade_effect, morph_effect, glitch_effect, pixel_shuffle_effect]
    selected = random.choice(effects)
    return selected(img1, img2, num_frames, intensity)

# --- UTILITY ---

def resize_and_crop(img, target_size):
    # img: PIL.Image, target_size: (w, h)
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    if img_ratio > target_ratio:
        # Immagine più larga, crop larghezza
        new_height = img.height
        new_width = int(new_height * target_ratio)
    else:
        # Immagine più alta, crop altezza
        new_width = img.width
        new_height = int(new_width / target_ratio)
    left = (img.width - new_width) // 2
    top = (img.height - new_height) // 2
    img_cropped = img.crop((left, top, left + new_width, top + new_height))
    img_resized = img_cropped.resize(target_size, Image.LANCZOS)
    return img_resized

# --- STREAMLIT UI ---

st.title("🎞️ Frame-to-Frame FX Multi-Image Generator")

uploaded_img1 = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"])
uploaded_img2 = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])

duration = st.slider("Video duration (seconds)", min_value=1, max_value=30, value=5)
fps = 24
num_frames = duration * fps

effect = st.selectbox("Choose effect", options=["Fade", "Morph", "Glitch", "Pixel Shuffle", "Random"])
intensity = st.selectbox("Choose intensity", options=["Soft", "Medium", "Hard"])
output_format = st.selectbox("Output format", options=["1:1", "9:16", "16:9"])

preview_enabled = st.checkbox("Show preview (small gif)", value=True)

if st.button("Generate Video"):

    if not uploaded_img1 or not uploaded_img2:
        st.error("Upload both images to proceed")
        st.stop()

    # Definisco dimensioni output in base al formato
    format_map = {
        "1:1": (512, 512),
        "9:16": (512, 910),
        "16:9": (910, 512)
    }
    target_size = format_map[output_format]

    # Carica immagini e converti a numpy array
    img1 = Image.open(uploaded_img1).convert("RGB")
    img2 = Image.open(uploaded_img2).convert("RGB")

    img1 = resize_and_crop(img1, target_size)
    img2 = resize_and_crop(img2, target_size)

    np_img1 = np.array(img1)
    np_img2 = np.array(img2)

    effect_funcs = {
        "Fade": fade_effect,
        "Morph": morph_effect,
        "Glitch": glitch_effect,
        "Pixel Shuffle": pixel_shuffle_effect,
        "Random": random_effect
    }

    func = effect_funcs[effect]

    st.info("Generating frames... Please wait.")
    progress_bar = st.progress(0)
    all_frames = []

    # Suddivido i frame per mostrare avanzamento
    step = max(num_frames // 100, 1)
    for i in range(0, num_frames, step):
        frames = func(np_img1, np_img2, step, intensity)
        all_frames.extend(frames)
        progress_bar.progress(min(i // step + 1, 100))

    # Output temporaneo video
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_filepath = tmp_file.name
    tmp_file.close()

    # Scrivo video con imageio (codec h264)
    writer = imageio.get_writer(tmp_filepath, fps=fps, codec="libx264", quality=8)

    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    st.success(f"Video generated successfully! Duration: {duration} seconds")

    if preview_enabled:
        # Preview: carico GIF leggera per non appesantire
        preview_gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
        imageio.mimsave(preview_gif_path, all_frames[:min(30, len(all_frames))], fps=10)
        st.image(preview_gif_path, caption="Preview (GIF)")

    # Offro download
    with open(tmp_filepath, "rb") as f:
        video_bytes = f.read()
    st.download_button(label="Download video", data=video_bytes, file_name="output_video.mp4", mime="video/mp4")

    # Rimuovo file temporanei
    os.remove(tmp_filepath)
    if preview_enabled:
        os.remove(preview_gif_path)
