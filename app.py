import numpy as np
from PIL import Image
import streamlit as st
import tempfile
import os
import math
from typing import List, Tuple

# --- Configurazione Streamlit ---
st.set_page_config(
    page_title="🎞️ Advanced Morphing", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("🔄 Morphing Multi-Immagine")

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
    "Personalizzato": None
}

# --- Funzioni Core ---
def calculate_frames(duration_sec: int, fps: int) -> int:
    """Calcola il numero di frame in base a durata e fps."""
    return int(duration_sec * fps)

def load_and_process_image(file, target_size: Tuple[int, int]) -> np.ndarray:
    """Carica e ridimensiona un'immagine."""
    img = Image.open(file).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)

def grid_warp_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Morphing avanzato con deformazione di griglia."""
    h, w = img1.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        # Distorsione dinamica
        distortion = 20 * math.sin(alpha * math.pi)  # Effetto a "onda"
        
        dx = distortion * np.sin(y/h * 2*math.pi + alpha * 2*math.pi)
        dy = distortion * np.cos(x/w * 2*math.pi + alpha * 2*math.pi)
        
        xn = np.clip(x + dx, 0, w-1)
        yn = np.clip(y + dy, 0, h-1)
        
        # Interpolazione avanzata
        morphed = np.zeros_like(img1)
        for c in range(3):
            morphed[..., c] = (
                img1[..., c] * (1 - alpha) + 
                img2[..., c] * alpha
            )
        
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def generate_transitions(images: List[np.ndarray], duration_sec: int, fps: int) -> List[np.ndarray]:
    """Genera tutte le transizioni tra le immagini."""
    total_frames = calculate_frames(duration_sec, fps)
    frames_per_transition = total_frames // (len(images) - 1)
    
    all_frames = []
    for i in range(len(images) - 1):
        transition = grid_warp_morph(images[i], images[i+1], frames_per_transition)
        all_frames.extend(transition)
    
    return all_frames

def save_video(frames: List[np.ndarray], path: str, fps: int) -> None:
    """Salva come GIF (sostituibile con codice per video MP4)."""
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000/fps),
        loop=0,
        optimize=True
    )

# --- Interfaccia Utente ---
def main():
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Impostazioni")
        
        # Selezione formato
        aspect_name = st.selectbox("Formato output", list(ASPECT_RATIOS.keys()))
        aspect_ratio = ASPECT_RATIOS[aspect_name]
        
        # Dimensioni personalizzate
        if aspect_ratio is None:
            custom_width = st.number_input("Larghezza", 100, 3840, 1920)
            custom_height = st.number_input("Altezza", 100, 2160, 1080)
            target_size = (custom_width, custom_height)
        else:
            base_res = st.selectbox("Risoluzione base", [480, 720, 1080, 1440], index=2)
            width_ratio, height_ratio = aspect_ratio
            if width_ratio >= height_ratio:  # Landscape o quadrato
                target_size = (int(base_res * width_ratio / height_ratio), base_res)
            else:  # Portrait
                target_size = (base_res, int(base_res * height_ratio / width_ratio))
        
        # Controlli temporali
        duration_sec = st.selectbox("Durata totale (secondi)", DURATION_OPTIONS, index=0)
        fps = st.selectbox("FPS", FPS_OPTIONS, index=1)
        
        st.markdown(f"**Frame totali:** {calculate_frames(duration_sec, fps)}")
    
    # Upload immagini
    uploaded_files = st.file_uploader(
        f"Carica {MIN_IMAGES}-{MAX_IMAGES} immagini",
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
            if st.button("🎬 Genera Video", type="primary"):
                with st.spinner(f"Generazione video ({duration_sec} secondi)..."):
                    # Genera tutti i frame
                    all_frames = generate_transitions(images, duration_sec, fps)
                    
                    # Salva temporaneamente
                    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                        save_video(all_frames, tmp.name, fps)
                        
                        # Mostra risultato
                        st.success("Completato!")
                        st.video(tmp.name)
                        
                        # Download
                        with open(tmp.name, "rb") as f:
                            st.download_button(
                                "💾 Scarica Video",
                                f.read(),
                                file_name=f"morphing_{duration_sec}s_{target_size[0]}x{target_size[1]}.gif",
                                mime="image/gif"
                            )
                    
                    # Cleanup
                    os.unlink(tmp.name)

if __name__ == "__main__":
    main()
