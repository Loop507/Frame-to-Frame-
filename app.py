import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from scipy import interpolate

# Configurazione pagina
st.set_page_config(
    page_title="🎞️ Image Morphing Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verifica Python
if not (3, 8) <= sys.version_info < (3, 11):
    st.error("Richiesto Python 3.8-3.10")
    st.stop()

# Effetti disponibili
EFFECTS = {
    "Morphing Avanzato": "advanced",
    "Dissolvenza": "fade",
    "Zoom": "zoom",
    "Glitch": "glitch"
}

def apply_effect(img1, img2, effect, alpha):
    """Applica l'effetto selezionato"""
    h, w = img1.shape[:2]
    
    if effect == "fade":
        return (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
    
    elif effect == "zoom":
        zoom = 1 + alpha
        matrix = cv2.getRotationMatrix2D((w//2, h//2), 0, zoom)
        return cv2.warpAffine(img1, matrix, (w, h))
    
    elif effect == "glitch":
        distorted = img1.copy()
        if alpha > 0.5:
            rows = np.random.randint(0, h, size=10)
            distorted[rows] = img2[rows]
        return distorted
    
    else:  # Morphing avanzato
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        morphed = np.zeros_like(img1)
        for c in range(3):  # RGB
            morphed[..., c] = cv2.remap(
                img1[..., c],
                (np.arange(w) + flow[..., 0] * alpha).astype(np.float32),
                (np.arange(h) + flow[..., 1] * alpha).astype(np.float32),
                cv2.INTER_LANCZOS4
            )
        return morphed

def main():
    st.title("🔄 Image Morphing Web App")
    
    with st.sidebar:
        st.header("⚙️ Impostazioni")
        effect = st.selectbox("Effetto", list(EFFECTS.keys()))
        duration = st.slider("Durata (sec)", 2, 10, 5)
        fps = st.slider("FPS", 15, 60, 24)
        
    uploaded_files = st.file_uploader(
        "Carica 2-5 immagini (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files and 2 <= len(uploaded_files) <= 5:
        if st.button("🎬 Genera Video"):
            with st.spinner("Elaborazione..."):
                progress_bar = st.progress(0)
                
                # Caricamento immagini
                images = []
                for i, file in enumerate(uploaded_files):
                    img = np.array(Image.open(file).convert("RGB"))
                    img = cv2.resize(img, (800, 600))
                    images.append(img)
                    progress_bar.progress((i + 1) / len(uploaded_files) / 3)
                
                # Generazione transizioni
                total_frames = int(duration * fps)
                frames_per_trans = max(10, total_frames // (len(images) - 1))
                frames = []
                
                for i in range(len(images) - 1):
                    for frame_idx in range(frames_per_trans):
                        alpha = frame_idx / frames_per_trans
                        frame = apply_effect(images[i], images[i+1], EFFECTS[effect], alpha)
                        frames.append(frame)
                        progress_bar.progress(0.33 + 0.67 * ((i * frames_per_trans + frame_idx) / total_frames))
                
                # Salvataggio
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(tmp_path, fourcc, fps, (800, 600))
                
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                
                # Output
                progress_bar.progress(1.0)
                st.success("Completato!")
                st.video(tmp_path)
                
                with open(tmp_path, "rb") as f:
                    st.download_button(
                        "💾 Scarica MP4",
                        f.read(),
                        file_name=f"morph_{effect}_{duration}s.mp4",
                        mime="video/mp4"
                    )
                
                os.unlink(tmp_path)

if __name__ == "__main__":
    import sys
    main()
