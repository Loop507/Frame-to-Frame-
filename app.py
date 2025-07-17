import streamlit as st
import numpy as np
from PIL import Image
import cv2
import imageio
import tempfile
import os
import random

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Multi-Image", layout="wide")

# --- Effetti ---

def fade_effect(img1, img2, num_frames):
    return [(img1 * (1 - alpha) + img2 * alpha).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

def morph_effect(img1, img2, num_frames):
    return [cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0).astype(np.uint8) for alpha in np.linspace(0, 1, num_frames)]

def glitch_effect(img1, img2, num_frames):
    frames = []
    h, w, _ = img1.shape
    for i in range(num_frames):
        frame = img1.copy()
        slice_height = h // 10
        for _ in range(random.randint(5, 15)):
            y = random.randint(0, h - slice_height)
            shift = random.randint(-20, 20)
            frame[y:y + slice_height, :] = np.roll(frame[y:y + slice_height, :], shift, axis=1)
        alpha = i / max(num_frames - 1, 1)
        frame = cv2.addWeighted(frame, 1 - alpha, img2, alpha, 0)
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_shuffle_effect(img1, img2, num_frames, intensity=10):
    frames = []
    h, w, _ = img1.shape
    for i in range(num_frames):
        frame = img1.copy()
        for _ in range(intensity):
            x = random.randint(0, w - 2)
            y = random.randint(0, h - 2)
            frame[y:y+2, x:x+2] = frame[y:y+2, x:x+2][::-1, ::-1]
        alpha = i / max(num_frames - 1, 1)
        frame = cv2.addWeighted(frame, 1 - alpha, img2, alpha, 0)
        frames.append(frame.astype(np.uint8))
    return frames

def random_effect(img1, img2, num_frames):
    effect_list = [fade_effect, morph_effect, glitch_effect, pixel_shuffle_effect]
    chosen = random.choice(effect_list)
    return chosen(img1, img2, num_frames)

# --- Controllo livelli effetti ---
def get_strength_frames(strength, base_frames):
    if strength == "Soft":
        return base_frames
    elif strength == "Medium":
        return base_frames * 2
    elif strength == "Hard":
        return base_frames * 3
    else:
        return base_frames

# --- Layout Streamlit ---

st.title("🎞️ Frame-to-Frame FX Multi-Image")

uploaded_files = st.file_uploader("Upload 2 to 10 images (JPEG/PNG)", accept_multiple_files=True, type=["jpg","jpeg","png"])

if uploaded_files:
    if not (2 <= len(uploaded_files) <= 10):
        st.error("Please upload between 2 and 10 images.")
    else:
        # Parametri
        effect = st.selectbox("Choose Effect", ["Fade", "Morph", "Glitch", "Pixel Shuffle", "Random"])
        strength = st.selectbox("Effect Strength", ["Soft", "Medium", "Hard"])
        duration = st.slider("Video Duration (seconds)", min_value=1, max_value=30, value=5)
        fps = 24

        res_option = st.selectbox("Output Resolution", ["1:1 (Square)", "9:16 (Vertical)", "16:9 (Horizontal)"])
        res_dict = {
            "1:1 (Square)": (720, 720),
            "9:16 (Vertical)": (540, 960),
            "16:9 (Horizontal)": (1280, 720)
        }
        out_w, out_h = res_dict[res_option]

        base_frames_per_transition = int(duration * fps / (len(uploaded_files) - 1))
        frames_per_transition = get_strength_frames(strength, base_frames_per_transition)

        # Effetto mappato
        effect_map = {
            "Fade": fade_effect,
            "Morph": morph_effect,
            "Glitch": glitch_effect,
            "Pixel Shuffle": pixel_shuffle_effect,
            "Random": random_effect,
        }

        all_frames = []
        # Preprocessing immagini (resize + convert to np.array)
        imgs = []
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            image = image.resize((out_w, out_h))
            imgs.append(np.array(image))

        # Generazione frame
        progress_bar = st.progress(0)
        total_transitions = len(imgs) - 1
        frame_count = 0
        total_frames = frames_per_transition * total_transitions

        for i in range(total_transitions):
            img1 = imgs[i]
            img2 = imgs[i+1]
            func = effect_map[effect]
            frames = func(img1, img2, frames_per_transition)
            all_frames.extend(frames)
            frame_count += frames_per_transition
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        # Salvataggio video
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_filepath = tmp_file.name
        tmp_file.close()

        try:
            writer = imageio.get_writer(tmp_filepath, fps=fps, codec="libx264", quality=8)
            for frame in all_frames:
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            st.error(f"Error during video creation: {e}")
            st.stop()

        st.success(f"Video generated! Duration: {duration}s, Resolution: {out_w}x{out_h}")

        # Preview leggera - mostra solo primo frame
        st.image(all_frames[0], caption="Preview first frame", use_container_width=True)

        # Download
        with open(tmp_filepath, "rb") as f:
            video_bytes = f.read()
        st.download_button(label="Download MP4 Video", data=video_bytes, file_name="output_video.mp4", mime="video/mp4")

        os.remove(tmp_filepath)
