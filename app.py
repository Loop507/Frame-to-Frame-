import numpy as np
from PIL import Image
import streamlit as st
import tempfile
import os
import cv2
from scipy import interpolate
import time
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
DURATION_OPTIONS = [5, 10, 15, 30]
FPS_OPTIONS = [24, 30, 60]

# --- Effetti Disponibili ---
EFFECTS = {
    "Morphing Avanzato": "advanced",
    "Dissolvenza": "fade", 
    "Zoom": "zoom",
    "Glitch": "glitch",
    "Pixel Art": "pixel",
    "Onda": "wave"
}

# --- Funzioni Effetti ---
def apply_effect(img1: np.ndarray, img2: np.ndarray, effect: str, alpha: float) -> np.ndarray:
    """Applica l'effetto selezionato"""
    if effect == "fade":
        return (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
    
    elif effect == "zoom":
        zoom = 1 + alpha
        h, w = img1.shape[:2]
        center = (w//2, h//2)
        matrix = cv2.getRotationMatrix2D(center, 0, zoom)
        return cv2.warpAffine(img1, matrix, (w,h))
    
    elif effect == "glitch":
        distorted = img1.copy()
        if alpha > 0.3:
            rows = np.random.randint(0, h, 10)
            distorted[rows] = img2[rows]
        return distorted
    
    elif effect == "pixel":
        if alpha < 0.5:
            size = int(10 * (1 - alpha*2))
            small = cv2.resize(img1, (w//size, h//size), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
        return img2
    
    elif effect == "wave":
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        distortion = 10 * np.sin(x/20 + alpha*10)
        xn = np.clip(x + distortion, 0, w-1)
        yn = np.clip(y + distortion, 0, h-1)
        return cv2.remap(img1, xn.astype(np.float32), yn.astype(np.float32), cv2.INTER_LINEAR)
    
    else:  # Advanced morphing
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        morphed = np.zeros_like(img1)
        for c in range(3):
            morphed[...,c] = cv2.remap(
                img1[...,c], 
                (np.arange(w) + flow[...,0]*alpha).astype(np.float32),
                (np.arange(h) + flow[...,1]*alpha).astype(np.float32),
                cv2.INTER_LANCZOS4
            )
        return morphed.astype(np.uint8)

# --- Interfaccia Utente ---
def main():
    with st.sidebar:
        st.header("⚙️ Impostazioni")
        
        # Controlli effetto
        effect = st.selectbox("Effetto", list(EFFECTS.keys()))
        duration = st.select_slider("Durata (sec)", options=DURATION_OPTIONS, value=5)
        fps = st.selectbox("FPS", FPS_OPTIONS, index=1)
        
        # Caricamento immagini
        uploaded_files = st.file_uploader(
            f"Carica {MIN_IMAGES}-{MAX_IMAGES} immagini",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True
        )

    # Generazione video
    if uploaded_files and len(uploaded_files) >= MIN_IMAGES:
        if st.button("🎬 Genera Video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preprocess immagini
            images = []
            for i, file in enumerate(uploaded_files[:MAX_IMAGES]):
                img = np.array(Image.open(file).convert("RGB"))
                images.append(img)
                progress_bar.progress((i+1)/len(uploaded_files[:MAX_IMAGES]) * 0.3)
                status_text.text(f"Caricamento immagini: {i+1}/{len(uploaded_files[:MAX_IMAGES])}")
            
            # Calcolo frame
            total_frames = int(duration * fps)
            frames = []
            
            # Generazione frame
            for i in range(len(images)-1):
                for frame_idx in range(total_frames // (len(images)-1)):
                    alpha = frame_idx / (total_frames // (len(images)-1))
                    frame = apply_effect(images[i], images[i+1], EFFECTS[effect], alpha)
                    frames.append(frame)
                    
                    # Aggiornamento progresso
                    progress = 0.3 + 0.7 * ((i * (total_frames//(len(images)-1)) + frame_idx) / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Generazione frame: {len(frames)}/{total_frames}\n"
                        f"Transizione: {i+1}/{len(images)-1}"
                    )
            
            # Salvataggio video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (images[0].shape[1], images[0].shape[0]))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            
            # Output
            progress_bar.progress(1.0)
            status_text.text("Completato!")
            st.video(tmp_path)
            
            # Download
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "💾 Scarica Video",
                    f.read(),
                    file_name=f"morph_{effect}_{duration}s.mp4",
                    mime="video/mp4"
                )
            
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
