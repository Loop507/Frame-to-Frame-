# morphing_app.py
import numpy as np
from PIL import Image
import streamlit as st
import tempfile
import os
import math
import cv2
from scipy import interpolate
from typing import List, Tuple

# --- Configurazione Streamlit ---
st.set_page_config(
    page_title="🎞️ Advanced Morphing Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("🔄 Morphing Multi-Immagine Pro")

# --- Costanti ---
MAX_IMAGES = 10
MIN_IMAGES = 2
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
DURATION_OPTIONS = [5, 10, 15, 30]  # Secondi
FPS_OPTIONS = [24, 30, 60]

# --- Preset Dimensioni ---
ASPECT_RATIOS = {
    "1:1 (Quadrato)": (1, 1),
    "9:16 (Verticale)": (9, 16),
    "16:9 (Orizzontale)": (16, 9),
    "4:3 (Classico)": (4, 3),
    "21:9 (Cinema)": (21, 9),
    "Personalizzato": None
}

# --- Funzioni Core con OpenCV e Scipy ---
def calculate_frames(duration_sec: int, fps: int) -> int:
    return int(duration_sec * fps)

def load_and_process_image(file, target_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    img = np.array(img)
    
    # Resize con OpenCV per mantenere qualità
    if img.shape[:2] != target_size[::-1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return img

def advanced_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Morphing avanzato con optical flow e interpolazione spline"""
    h, w = img1.shape[:2]
    
    # Calcola optical flow con OpenCV
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Crea griglia di coordinate
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        # Interpolazione del flusso ottico
        interp_flow = flow * alpha
        xn = x + interp_flow[..., 0]
        yn = y + interp_flow[..., 1]
        
        # Interpolazione spline (Scipy)
        morphed = np.zeros_like(img1)
        for c in range(3):
            spline = interpolate.RectBivariateSpline(
                np.arange(h), np.arange(w), img1[..., c]
            )
            morphed[..., c] = spline.ev(yn, xn)
            
            spline = interpolate.RectBivariateSpline(
                np.arange(h), np.arange(w), img2[..., c]
            )
            morphed[..., c] = morphed[..., c] * (1 - alpha) + spline.ev(yn, xn) * alpha
        
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def generate_transitions(images: List[np.ndarray], duration_sec: int, fps: int) -> List[np.ndarray]:
    total_frames = calculate_frames(duration_sec, fps)
    frames_per_transition = max(10, total_frames // (len(images) - 1))
    
    all_frames = []
    for i in range(len(images) - 1):
        with st.spinner(f"Generando transizione {i+1}/{len(images)-1}..."):
            transition = advanced_morph(images[i], images[i+1], frames_per_transition)
            all_frames.extend(transition)
    
    return all_frames

def save_video(frames: List[np.ndarray], path: str, fps: int, target_size: Tuple[int, int]) -> None:
    """Salva come MP4 con OpenCV"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, target_size)
    
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()

# --- Interfaccia Utente ---
def main():
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Impostazioni Pro")
        
        # Selezione formato
        aspect_name = st.selectbox("Formato output", list(ASPECT_RATIOS.keys()))
        aspect_ratio = ASPECT_RATIOS[aspect_name]
        
        # Dimensioni
        if aspect_ratio is None:
            custom_width = st.number_input("Larghezza", 100, 3840, 1920)
            custom_height = st.number_input("Altezza", 100, 2160, 1080)
            target_size = (custom_width, custom_height)
        else:
            base_res = st.selectbox("Risoluzione base", [480, 720, 1080, 1440, 2160], index=2)
            width_ratio, height_ratio = aspect_ratio
            if width_ratio >= height_ratio:
                target_size = (int(base_res * width_ratio / height_ratio), base_res)
            else:
                target_size = (base_res, int(base_res * height_ratio / width_ratio))
        
        # Controlli temporali
        duration_sec = st.select_slider("Durata totale", options=DURATION_OPTIONS, value=5)
        fps = st.selectbox("Frame rate (FPS)", FPS_OPTIONS, index=1)
        
        st.markdown(f"""
        **Info tecniche:**
        - Frame totali: {calculate_frames(duration_sec, fps)}
        - Risoluzione: {target_size[0]}x{target_size[1]}
        - Formato: {aspect_name}
        """)
    
    # Upload immagini
    uploaded_files = st.file_uploader(
        f"Carica {MIN_IMAGES}-{MAX_IMAGES} immagini (ordine importante)",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) >= MIN_IMAGES:
        # Processa immagini
        images = []
        cols = st.columns(min(len(uploaded_files), 5))
        
        for i, file in enumerate(uploaded_files[:MAX_IMAGES]):
            with cols[i % 5]:
                try:
                    img = load_and_process_image(file, target_size)
                    images.append(img)
                    st.image(img, caption=f"Img {i+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Errore nell'immagine {i+1}: {str(e)}")
        
        if len(images) >= MIN_IMAGES:
            if st.button("🎬 Genera Video Pro", type="primary"):
                with st.spinner(f"Generazione video ({duration_sec}s)..."):
                    # Genera tutti i frame
                    all_frames = generate_transitions(images, duration_sec, fps)
                    
                    # Salva temporaneamente
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    save_video(all_frames, tmp_path, fps, target_size)
                    
                    # Mostra risultato
                    st.success(f"Video generato con successo! ({len(all_frames)} frame)")
                    st.video(tmp_path)
                    
                    # Download
                    with open(tmp_path, "rb") as f:
                        st.download_button(
                            "💾 Scarica MP4",
                            f.read(),
                            file_name=f"morphing_{duration_sec}s_{target_size[0]}x{target_size[1]}.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    os.unlink(tmp_path)

if __name__ == "__main__":
    main()
