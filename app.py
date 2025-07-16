import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from moviepy.editor import ImageSequenceClip
import tempfile
import io
import base64
import random
from scipy.ndimage import gaussian_filter, rotate, map_coordinates

# === Funzioni effetti ===

def load_image(image_file, size):
    img = Image.open(image_file).convert("RGB")
    return np.array(img.resize(size)).astype(np.uint8)

def fade(img1, img2, frames):
    return [(img1 * (1 - i / frames) + img2 * (i / frames)).astype(np.uint8) for i in range(frames)]

def rgb_shift(img, shift):
    b, g, r = cv2.split(img)
    r = np.roll(r, shift, axis=1)
    return cv2.merge([b, g, r])

def tv_noise(img, intensity=30):
    noise = np.random.randint(-intensity, intensity, img.shape, dtype=np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def swirl(img, strength=1.5):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + strength * r / max(h, w)
    x_new = (r * np.cos(theta) + cx).astype(np.float32)
    y_new = (r * np.sin(theta) + cy).astype(np.float32)
    warped = np.stack([map_coordinates(img[..., c], [y_new, x_new], order=1, mode='reflect') for c in range(3)], axis=-1)
    return warped.astype(np.uint8)

def morph(img1, img2, frames):
    return fade(img1, img2, frames)

def slide(img1, img2, frames, direction="horizontal"):
    h, w = img1.shape[:2]
    out = []
    for i in range(frames):
        alpha = i / frames
        offset = int(w * alpha) if direction == "horizontal" else int(h * alpha)
        frame = img1.copy()
        if direction == "horizontal":
            frame[:, offset:] = img2[:, :w - offset]
        else:
            frame[offset:, :] = img2[:h - offset, :]
        out.append(frame)
    return out

def apply_filter(img, filter_type):
    if filter_type == "blur":
        return cv2.GaussianBlur(img, (7, 7), 0)
    if filter_type == "edges":
        return cv2.Canny(img, 100, 200)
    if filter_type == "gray":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

# === Streamlit App ===
st.set_page_config(layout="wide")
st.title("🎥 Frame-to-Frame Effects Generator")

uploaded_files = st.file_uploader("📸 Carica le immagini (minimo 2)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
effect = st.selectbox("🎨 Seleziona effetto", ["fade", "morph", "rgb shift", "tv noise", "swirl", "slide horizontal", "slide vertical"])
filter_opt = st.selectbox("🌈 Filtro immagine", ["none", "blur", "edges", "gray"])
fps = st.slider("🎞️ FPS (frames per second)", 5, 60, 24)
frames_per_transition = st.slider("⏱️ Durata transizione (frame)", 5, 120, 30)
format_opt = st.selectbox("📐 Formato video", ["1:1", "16:9", "9:16"])

if uploaded_files and len(uploaded_files) >= 2:
    w, h = 512, 512
    if format_opt == "16:9":
        w, h = 640, 360
    elif format_opt == "9:16":
        w, h = 360, 640

    images = [apply_filter(load_image(file, (w, h)), filter_opt) for file in uploaded_files]
    all_frames = []
    progress = st.progress(0)

    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        if effect == "fade":
            transition = fade(img1, img2, frames_per_transition)
        elif effect == "morph":
            transition = morph(img1, img2, frames_per_transition)
        elif effect == "rgb shift":
            transition = [rgb_shift(img1, i % 10) for i in range(frames_per_transition)]
        elif effect == "tv noise":
            transition = [tv_noise(img1) for _ in range(frames_per_transition)]
        elif effect == "swirl":
            transition = [swirl(img1) for _ in range(frames_per_transition)]
        elif effect == "slide horizontal":
            transition = slide(img1, img2, frames_per_transition, "horizontal")
        elif effect == "slide vertical":
            transition = slide(img1, img2, frames_per_transition, "vertical")
        else:
            transition = fade(img1, img2, frames_per_transition)
        all_frames.extend(transition)
        progress.progress((i + 1) / (len(images) - 1))

    progress.empty()
    st.success("✅ Transizioni completate!")

    clip = ImageSequenceClip([frame for frame in all_frames], fps=fps)
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    clip.write_videofile(temp_video.name, codec="libx264", audio=False)

    with open(temp_video.name, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)
    st.download_button("📁 Scarica video", data=video_bytes, file_name="transizione_video.mp4")
else:
    st.info("Carica almeno 2 immagini per iniziare")
