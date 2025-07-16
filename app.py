import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import tempfile
import gc
from typing import List, Tuple, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🎞️ Frame-to-Frame FX Multi-Image", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Costanti ---
MAX_IMAGES = 5
MAX_DIMENSION = 1920
MIN_DIMENSION = 100
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]
DEFAULT_FPS = 30
DEFAULT_FRAMES_PER_TRANSITION = 60

# --- Utility Functions ---

def validate_image(img: Image.Image) -> bool:
    """Valida se l'immagine è utilizzabile."""
    if img is None:
        return False
    
    width, height = img.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return False
    
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        return False
    
    return True

def calculate_optimal_size(images: List[Image.Image], max_size: int = 1280) -> Tuple[int, int]:
    """Calcola la dimensione ottimale mantenendo le proporzioni."""
    if not images:
        return (640, 480)
    
    # Trova le dimensioni medie
    avg_width = sum(img.size[0] for img in images) / len(images)
    avg_height = sum(img.size[1] for img in images) / len(images)
    
    # Mantieni proporzioni
    aspect_ratio = avg_width / avg_height
    
    if avg_width > avg_height:
        width = min(max_size, int(avg_width))
        height = int(width / aspect_ratio)
    else:
        height = min(max_size, int(avg_height))
        width = int(height * aspect_ratio)
    
    # Assicurati che siano pari (per compatibilità video)
    width = width - (width % 2)
    height = height - (height % 2)
    
    return (width, height)

def preprocess_images(uploaded_files, target_size: Tuple[int, int]) -> List[np.ndarray]:
    """Preprocessa le immagini con gestione errori migliorata."""
    processed_images = []
    
    for i, file in enumerate(uploaded_files[:MAX_IMAGES]):
        try:
            img = Image.open(file).convert('RGB')
            
            if not validate_image(img):
                st.warning(f"Immagine {i+1} non valida (dimensioni troppo piccole/grandi)")
                continue
            
            # Ridimensiona mantenendo qualità
            img_resized = img.resize(target_size, Image.LANCZOS)
            processed_images.append(np.array(img_resized, dtype=np.uint8))
            
        except Exception as e:
            st.error(f"Errore nel processamento dell'immagine {i+1}: {str(e)}")
            continue
    
    return processed_images

# --- Effetti Migliorati ---

def fade_effect(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto dissolvenza ottimizzato."""
    frames = []
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = (img1_f * (1 - alpha) + img2_f * alpha).astype(np.uint8)
        frames.append(frame)
    
    return frames

def zoom_effect(img1: np.ndarray, img2: np.ndarray, num_frames: int, zoom_factor: float = 1.3) -> List[np.ndarray]:
    """Effetto zoom migliorato con interpolazione."""
    frames = []
    h, w = img1.shape[:2]
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        scale = 1 + (zoom_factor - 1) * alpha
        
        # Calcola nuove dimensioni
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Ridimensiona con interpolazione di alta qualità
        img1_pil = Image.fromarray(img1).resize((new_w, new_h), Image.LANCZOS)
        img2_pil = Image.fromarray(img2).resize((new_w, new_h), Image.LANCZOS)
        
        img1_scaled = np.array(img1_pil)
        img2_scaled = np.array(img2_pil)
        
        # Centra l'immagine
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        
        # Estrai la porzione centrale
        img1_crop = img1_scaled[offset_y:offset_y+h, offset_x:offset_x+w]
        img2_crop = img2_scaled[offset_y:offset_y+h, offset_x:offset_x+w]
        
        # Blending
        frame = (img1_crop.astype(np.float32) * (1 - alpha) + 
                img2_crop.astype(np.float32) * alpha).astype(np.uint8)
        frames.append(frame)
    
    return frames

def pixel_swap_random(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Pixel swap casuale ottimizzato."""
    frames = []
    h, w, c = img1.shape
    total_pixels = h * w
    
    # Pre-genera coordinate casuali
    np.random.seed(42)  # Per riproducibilità
    all_coords = np.random.permutation(total_pixels)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swapped = int(total_pixels * alpha)
        
        frame = img1.copy()
        
        # Usa coordinate pre-generate
        coords_to_swap = all_coords[:num_swapped]
        ys, xs = np.divmod(coords_to_swap, w)
        frame[ys, xs] = img2[ys, xs]
        
        frames.append(frame)
    
    return frames

def pixel_swap_blocks(img1: np.ndarray, img2: np.ndarray, num_frames: int, block_size: int = 16) -> List[np.ndarray]:
    """Pixel swap a blocchi con fix del loop infinito."""
    frames = []
    h, w, c = img1.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return [img1] * num_frames
    
    # Pre-genera ordine dei blocchi
    np.random.seed(42)
    all_blocks = [(by, bx) for by in range(blocks_h) for bx in range(blocks_w)]
    np.random.shuffle(all_blocks)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_to_swap = min(int(total_blocks * alpha), total_blocks)
        
        frame = img1.copy()
        
        # Usa blocchi pre-ordinati
        for j in range(num_to_swap):
            by, bx = all_blocks[j]
            y_start = by * block_size
            y_end = min(y_start + block_size, h)
            x_start = bx * block_size
            x_end = min(x_start + block_size, w)
            
            frame[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
        
        frames.append(frame)
    
    return frames

def pixel_swap_wave(img1: np.ndarray, img2: np.ndarray, num_frames: int, wave_amplitude: float = 20) -> List[np.ndarray]:
    """Effetto onda migliorato."""
    frames = []
    h, w, c = img1.shape
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        wave_pos = int(w * alpha)
        
        frame = img1.copy()
        
        for y in range(h):
            wave_offset = int(wave_amplitude * np.sin(y * 0.05 + i * 0.2))
            x_threshold = max(0, min(w, wave_pos + wave_offset))
            
            if x_threshold > 0:
                frame[y, :x_threshold] = img2[y, :x_threshold]
        
        frames.append(frame)
    
    return frames

def pixel_swap_spiral(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto spirale ottimizzato."""
    frames = []
    h, w, c = img1.shape
    center_y, center_x = h // 2, w // 2
    
    # Pre-calcola coordinate spirale
    spiral_coords = []
    max_radius = int(np.sqrt(center_x**2 + center_y**2)) + 1
    
    for radius in range(max_radius):
        if radius == 0:
            spiral_coords.append((center_y, center_x))
        else:
            steps = max(8, radius * 6)
            for angle in np.linspace(0, 2*np.pi, steps, endpoint=False):
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                if 0 <= x < w and 0 <= y < h:
                    spiral_coords.append((y, x))
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swapped = int(len(spiral_coords) * alpha)
        
        frame = img1.copy()
        
        for j in range(min(num_swapped, len(spiral_coords))):
            y, x = spiral_coords[j]
            frame[y, x] = img2[y, x]
        
        frames.append(frame)
    
    return frames

def create_video(frames: List[np.ndarray], output_path: str, fps: int = DEFAULT_FPS) -> bool:
    """Crea video con codec migliorato."""
    if not frames:
        return False
    
    try:
        h, w, _ = frames[0].shape
        
        # Usa codec più compatibile
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        if not out.isOpened():
            # Fallback a codec meno efficiente ma più supportato
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()
        return True
        
    except Exception as e:
        logger.error(f"Errore nella creazione del video: {e}")
        return False

# --- UI Migliorata ---

def main():
    st.title("🎞️ Frame-to-Frame FX Multi-Image")
    st.markdown("### Genera video con transizioni animate tra 2-5 immagini")
    
    # Sidebar per controlli
    with st.sidebar:
        st.header("⚙️ Impostazioni")
        
        effect = st.selectbox(
            "🎨 Effetto di transizione",
            [
                "Dissolvenza (fade)",
                "Zoom",
                "Pixel casuali",
                "Blocchi",
                "Onda",
                "Spirale"
            ],
            help="Seleziona l'effetto per le transizioni tra immagini"
        )
        
        fps = st.slider("🎚 FPS", 5, 60, DEFAULT_FPS, help="Frame per secondo del video")
        frames_per_transition = st.slider(
            "⏳ Frame per transizione", 
            10, 300, 
            DEFAULT_FRAMES_PER_TRANSITION,
            help="Numero di frame per ogni transizione"
        )
        
        if effect == "Blocchi":
            block_size = st.slider("📦 Dimensione blocchi", 4, 64, 16, step=4)
        else:
            block_size = 16
        
        st.info(f"📊 Stima durata: {len(st.session_state.get('uploaded_files', [])) * frames_per_transition / fps:.1f}s")
    
    # Upload immagini
    uploaded_files = st.file_uploader(
        "📁 Carica immagini (2-5)",
        accept_multiple_files=True,
        type=SUPPORTED_FORMATS,
        help=f"Formati supportati: {', '.join(SUPPORTED_FORMATS)}"
    )
    
    # Salva in session state
    if uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files
    
    # Mostra preview
    if uploaded_files:
        st.subheader("🖼️ Anteprima immagini")
        
        if len(uploaded_files) > MAX_IMAGES:
            st.warning(f"Verranno utilizzate solo le prime {MAX_IMAGES} immagini")
        
        if len(uploaded_files) < 2:
            st.error("Carica almeno 2 immagini per generare il video!")
            return
        
        # Mostra grid di anteprime
        cols = st.columns(min(len(uploaded_files), 5))
        preview_images = []
        
        for i, file in enumerate(uploaded_files[:MAX_IMAGES]):
            try:
                img = Image.open(file)
                preview_images.append(img)
                with cols[i]:
                    st.image(img, caption=f"Immagine {i+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Errore nel caricamento dell'immagine {i+1}: {e}")
        
        # Calcola dimensioni ottimali
        if preview_images:
            target_size = calculate_optimal_size(preview_images)
            st.info(f"🎯 Dimensioni video: {target_size[0]}x{target_size[1]}")
    
    # Genera video
    if st.button("🎬 Genera Video", type="primary"):
        if not uploaded_files or len(uploaded_files) < 2:
            st.error("Carica almeno 2 immagini!")
            return
        
        with st.spinner("🔄 Elaborazione in corso..."):
            try:
                # Preprocessa immagini
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📥 Caricamento immagini...")
                progress_bar.progress(0.1)
                
                # Calcola dimensioni ottimali
                pil_images = []
                for file in uploaded_files[:MAX_IMAGES]:
                    pil_images.append(Image.open(file))
                
                target_size = calculate_optimal_size(pil_images)
                processed_images = preprocess_images(uploaded_files, target_size)
                
                if len(processed_images) < 2:
                    st.error("Non sono state processate abbastanza immagini valide!")
                    return
                
                progress_bar.progress(0.3)
                status_text.text("🎨 Generazione effetti...")
                
                # Seleziona funzione effetto
                effect_functions = {
                    "Dissolvenza (fade)": fade_effect,
                    "Zoom": zoom_effect,
                    "Pixel casuali": pixel_swap_random,
                    "Blocchi": lambda img1, img2, num_frames: pixel_swap_blocks(img1, img2, num_frames, block_size),
                    "Onda": pixel_swap_wave,
                    "Spirale": pixel_swap_spiral
                }
                
                effect_func = effect_functions[effect]
                all_frames = []
                n_imgs = len(processed_images)
                
                # Genera transizioni
                for i in range(n_imgs):
                    img1 = processed_images[i]
                    img2 = processed_images[(i + 1) % n_imgs]
                    
                    transition_frames = effect_func(img1, img2, frames_per_transition)
                    all_frames.extend(transition_frames)
                    
                    progress = 0.3 + (0.6 * (i + 1) / n_imgs)
                    progress_bar.progress(progress)
                    status_text.text(f"🎬 Transizione {i+1}/{n_imgs}")
                
                progress_bar.progress(0.9)
                status_text.text("💾 Creazione video...")
                
                # Crea video temporaneo
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                success = create_video(all_frames, tmp_path, fps)
                
                if success and os.path.exists(tmp_path):
                    progress_bar.progress(1.0)
                    status_text.text("✅ Video completato!")
                    
                    # Mostra informazioni
                    file_size = os.path.getsize(tmp_path)
                    duration = len(all_frames) / fps
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Durata", f"{duration:.1f}s")
                    with col2:
                        st.metric("📏 Dimensioni", f"{target_size[0]}x{target_size[1]}")
                    with col3:
                        st.metric("📦 Dimensione file", f"{file_size/1024/1024:.1f} MB")
                    
                    # Download
                    with open(tmp_path, "rb") as f:
                        st.download_button(
                            "📥 Scarica Video MP4",
                            f.read(),
                            file_name=f"frame_to_frame_{effect.lower().replace(' ', '_')}.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    # Libera memoria
                    del all_frames, processed_images
                    gc.collect()
                    
                else:
                    st.error("❌ Errore nella generazione del video")
                    
            except Exception as e:
                st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
                logger.error(f"Errore elaborazione: {e}")

if __name__ == "__main__":
    main()
