import numpy as np
from PIL import Image
import streamlit as st
import tempfile
import os
import cv2
from typing import List, Tuple

# --- Configurazione Streamlit ---
st.set_page_config(page_title="🎞️ Morphing Base", layout="wide")
st.title("🔄 Morphing tra Immagini")

# --- Costanti ---
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
DEFAULT_FPS = 24
DEFAULT_FRAMES = 30

# --- Funzioni Core ---
def load_image(uploaded_file) -> np.ndarray:
    """Carica un'immagine come array numpy."""
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def resize_to_target(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Ridimensiona mantenendo aspect ratio con crop centrale."""
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

def grid_warp_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Morphing basato su deformazione di griglia."""
    h, w = img1.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        distortion = 15 * alpha * (1 - alpha)
        
        dx = distortion * np.sin(y / h * 2 * np.pi + alpha * np.pi)
        dy = distortion * np.cos(x / w * 2 * np.pi + alpha * np.pi)
        
        xn = np.clip(x + dx, 0, w - 1)
        yn = np.clip(y + dy, 0, h - 1)
        
        xf = np.floor(xn).astype(int)
        yf = np.floor(yn).astype(int)
        xc = np.clip(xf + 1, 0, w - 1)
        yc = np.clip(yf + 1, 0, h - 1)
        
        a = xn - xf
        b = yn - yf
        
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = img1[..., c] * (1 - alpha) + img2[..., c] * alpha
            top = (1 - a) * chan[yf, xf] + a * chan[yf, xc]
            bottom = (1 - a) * chan[yc, xf] + a * chan[yc, xc]
            morphed[..., c] = (1 - b) * top + b * bottom
        
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def save_as_gif(frames: List[np.ndarray], path: str, fps: int) -> None:
    """Salva i frame come GIF."""
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )

def save_as_mp4(frames: List[np.ndarray], path: str, fps: int) -> None:
    """Salva i frame come video MP4."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()

# --- Interfaccia Streamlit ---
def main():
    st.sidebar.header("⚙️ Impostazioni")
    target_width = st.sidebar.number_input("Larghezza", 100, 1920, 800)
    target_height = st.sidebar.number_input("Altezza", 100, 1080, 600)
    fps = st.sidebar.number_input("FPS", 5, 60, DEFAULT_FPS)
    num_frames = st.sidebar.number_input("Frame", 10, 300, DEFAULT_FRAMES)
    
    uploaded_files = st.file_uploader(
        "Carica 2 immagini",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        target_size = (target_width, target_height)
        
        with st.spinner("Elaborazione in corso..."):
            img1 = resize_to_target(load_image(uploaded_files[0]), target_size)
            img2 = resize_to_target(load_image(uploaded_files[1]), target_size)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img1, caption="Immagine 1", use_column_width=True)
            with col2:
                st.image(img2, caption="Immagine 2", use_column_width=True)

            frames = grid_warp_morph(img1, img2, num_frames)

            with tempfile.TemporaryDirectory() as tmpdir:
                gif_path = os.path.join(tmpdir, "morphing.gif")
                mp4_path = os.path.join(tmpdir, "morphing.mp4")

                save_as_gif(frames, gif_path, fps)
                save_as_mp4(frames, mp4_path, fps)

                st.success("Morphing completato!")
                st.image(gif_path, use_column_width=True)

                with open(gif_path, "rb") as f:
                    st.download_button("Scarica GIF", f.read(), file_name="morphing.gif", mime="image/gif")
                with open(mp4_path, "rb") as f:
                    st.download_button("Scarica MP4", f.read(), file_name="morphing.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
