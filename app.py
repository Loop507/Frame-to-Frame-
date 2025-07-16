import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import tempfile
import gc
from typing import List, Tuple, Optional
import logging
import base64

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

# Risoluzioni predefinite
ASPECT_RATIOS = {
    "Quadrato (1:1)": (1, 1),
    "Landscape (16:9)": (16, 9),
    "Portrait (9:16)": (9, 16),
    "Cinema (21:9)": (21, 9),
    "Classico (4:3)": (4, 3),
    "Portrait (3:4)": (3, 4),
    "Auto (basato su immagini)": None
}

QUALITY_PRESETS = {
    "Bassa (480p)": 480,
    "Media (720p)": 720,
    "Alta (1080p)": 1080,
    "Ultra (1440p)": 1440
}

# --- Utility Functions ---

def calculate_resolution(aspect_ratio: Tuple[int, int], max_height: int) -> Tuple[int, int]:
    """Calcola risoluzione basata su aspect ratio e qualità."""
    if aspect_ratio is None:
        return None
    
    width_ratio, height_ratio = aspect_ratio
    
    # Calcola dimensioni mantenendo l'aspect ratio
    if width_ratio >= height_ratio:
        # Landscape o quadrato
        height = max_height
        width = int((height * width_ratio) / height_ratio)
    else:
        # Portrait
        width = max_height
        height = int((width * height_ratio) / width_ratio)
    
    # Assicurati che siano pari
    width = width - (width % 2)
    height = height - (height % 2)
    
    return (width, height)

def smart_resize_with_crop(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Ridimensiona l'immagine mantenendo proporzioni e facendo crop intelligente."""
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    # Calcola scale factors
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    # Usa lo scale maggiore per riempire completamente il target
    scale = max(scale_x, scale_y)
    
    # Calcola nuove dimensioni
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Ridimensiona
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Calcola crop per centrare
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop
    img_cropped = img_resized.crop((left, top, right, bottom))
    
    return img_cropped

def preprocess_images(uploaded_files, target_size: Tuple[int, int]) -> List[np.ndarray]:
    """Preprocessa le immagini con smart resize."""
    processed_images = []
    
    for i, file in enumerate(uploaded_files[:MAX_IMAGES]):
        try:
            img = Image.open(file).convert('RGB')
            
            # Smart resize con crop
            img_resized = smart_resize_with_crop(img, target_size)
            processed_images.append(np.array(img_resized, dtype=np.uint8))
            
        except Exception as e:
            st.error(f"Errore nel processamento dell'immagine {i+1}: {str(e)}")
            continue
    
    return processed_images

def calculate_optical_flow_keypoints(img1: np.ndarray, img2: np.ndarray, num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Calcola keypoints per morphing usando optical flow semplificato."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Trova corners nell'immagine 1
    corners1 = cv2.goodFeaturesToTrack(gray1, num_points, 0.01, 10)
    
    if corners1 is not None:
        # Calcola optical flow
        corners2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners1, None)
        
        # Filtra punti validi
        good_corners1 = corners1[status == 1]
        good_corners2 = corners2[status == 1]
        
        return good_corners1, good_corners2
    
    return np.array([]), np.array([])

# --- Nuovi Effetti ---

def morphing_effect(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto morphing semplificato usando optical flow."""
    frames = []
    h, w = img1.shape[:2]
    
    # Calcola keypoints
    points1, points2 = calculate_optical_flow_keypoints(img1, img2, 200)
    
    # Se non ci sono punti sufficienti, fallback a fade
    if len(points1) < 10:
        return fade_effect(img1, img2, num_frames)
    
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Interpolazione semplice dei punti
        frame = img1_f * (1 - alpha) + img2_f * alpha
        
        # Aggiungi deformazione locale basata sui keypoints
        if len(points1) > 0:
            # Crea una mappa di deformazione semplice
            for j in range(min(len(points1), 50)):  # Usa solo primi 50 punti
                p1 = points1[j][0]
                p2 = points2[j][0]
                
                # Interpolazione della posizione
                current_pos = p1 * (1 - alpha) + p2 * alpha
                
                # Applica deformazione locale molto sottile
                center_x, center_y = int(current_pos[0]), int(current_pos[1])
                radius = 15
                
                y_min = max(0, center_y - radius)
                y_max = min(h, center_y + radius)
                x_min = max(0, center_x - radius)
                x_max = min(w, center_x + radius)
                
                if y_max > y_min and x_max > x_min:
                    # Applica blending locale più forte
                    local_alpha = alpha * 1.2
                    local_alpha = min(1.0, local_alpha)
                    
                    frame[y_min:y_max, x_min:x_max] = (
                        img1_f[y_min:y_max, x_min:x_max] * (1 - local_alpha) + 
                        img2_f[y_min:y_max, x_min:x_max] * local_alpha
                    )
        
        frames.append(frame.astype(np.uint8))
    
    return frames

def glitch_lines_effect(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto glitch con linee orizzontali."""
    frames = []
    h, w, c = img1.shape
    
    # Pre-genera pattern di glitch
    np.random.seed(42)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Base frame con fade
        base_frame = (img1.astype(np.float32) * (1 - alpha) + 
                     img2.astype(np.float32) * alpha).astype(np.uint8)
        
        # Aggiungi glitch lines
        glitch_frame = base_frame.copy()
        
        # Numero di linee glitch basato sulla progressione
        num_lines = int(20 * np.sin(alpha * np.pi))
        
        for _ in range(num_lines):
            # Scegli riga casuale
            line_y = np.random.randint(0, h)
            line_height = np.random.randint(1, 8)
            
            # Scegli sorgente (img1 o img2)
            if np.random.random() < 0.5:
                source_img = img1
            else:
                source_img = img2
            
            # Applica offset orizzontale
            offset = np.random.randint(-20, 21)
            
            # Copia la linea con offset
            y_end = min(line_y + line_height, h)
            
            if offset > 0:
                # Shift a destra
                glitch_frame[line_y:y_end, offset:] = source_img[line_y:y_end, :w-offset]
            elif offset < 0:
                # Shift a sinistra
                glitch_frame[line_y:y_end, :w+offset] = source_img[line_y:y_end, -offset:]
            
            # Aggiungi distorsione colore
            if np.random.random() < 0.3:
                # Separa canali RGB
                glitch_frame[line_y:y_end, :, 0] = source_img[line_y:y_end, :, 0]  # R
                if np.random.random() < 0.5:
                    glitch_frame[line_y:y_end, :, 1] = img2[line_y:y_end, :, 1]  # G
                if np.random.random() < 0.5:
                    glitch_frame[line_y:y_end, :, 2] = img1[line_y:y_end, :, 2]  # B
        
        frames.append(glitch_frame)
    
    return frames

def digital_noise_effect(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto rumore digitale."""
    frames = []
    h, w, c = img1.shape
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Base transition
        base_frame = (img1.astype(np.float32) * (1 - alpha) + 
                     img2.astype(np.float32) * alpha).astype(np.uint8)
        
        # Aggiungi rumore digitale
        noise_intensity = np.sin(alpha * np.pi) * 0.3
        
        if noise_intensity > 0:
            # Genera rumore
            noise = np.random.randint(-int(noise_intensity * 50), 
                                    int(noise_intensity * 50) + 1, 
                                    size=(h, w, c))
            
            # Applica rumore con clipping
            noisy_frame = base_frame.astype(np.int16) + noise
            noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
            
            frames.append(noisy_frame)
        else:
            frames.append(base_frame)
    
    return frames

# --- Effetti Esistenti (mantenuti) ---

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
    """Effetto zoom migliorato."""
    frames = []
    h, w = img1.shape[:2]
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        scale = 1 + (zoom_factor - 1) * alpha
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        img1_pil = Image.fromarray(img1).resize((new_w, new_h), Image.LANCZOS)
        img2_pil = Image.fromarray(img2).resize((new_w, new_h), Image.LANCZOS)
        
        img1_scaled = np.array(img1_pil)
        img2_scaled = np.array(img2_pil)
        
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        
        img1_crop = img1_scaled[offset_y:offset_y+h, offset_x:offset_x+w]
        img2_crop = img2_scaled[offset_y:offset_y+h, offset_x:offset_x+w]
        
        frame = (img1_crop.astype(np.float32) * (1 - alpha) + 
                img2_crop.astype(np.float32) * alpha).astype(np.uint8)
        frames.append(frame)
    
    return frames

def pixel_swap_random(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Pixel swap casuale."""
    frames = []
    h, w, c = img1.shape
    total_pixels = h * w
    
    np.random.seed(42)
    all_coords = np.random.permutation(total_pixels)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swapped = int(total_pixels * alpha)
        
        frame = img1.copy()
        coords_to_swap = all_coords[:num_swapped]
        ys, xs = np.divmod(coords_to_swap, w)
        frame[ys, xs] = img2[ys, xs]
        
        frames.append(frame)
    
    return frames

def pixel_swap_blocks(img1: np.ndarray, img2: np.ndarray, num_frames: int, block_size: int = 16) -> List[np.ndarray]:
    """Pixel swap a blocchi."""
    frames = []
    h, w, c = img1.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return [img1] * num_frames
    
    np.random.seed(42)
    all_blocks = [(by, bx) for by in range(blocks_h) for bx in range(blocks_w)]
    np.random.shuffle(all_blocks)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_to_swap = min(int(total_blocks * alpha), total_blocks)
        
        frame = img1.copy()
        
        for j in range(num_to_swap):
            by, bx = all_blocks[j]
            y_start = by * block_size
            y_end = min(y_start + block_size, h)
            x_start = bx * block_size
            x_end = min(x_start + block_size, w)
            
            frame[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
        
        frames.append(frame)
    
    return frames

def pixel_swap_wave(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto onda."""
    frames = []
    h, w, c = img1.shape
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        wave_pos = int(w * alpha)
        
        frame = img1.copy()
        
        for y in range(h):
            wave_offset = int(20 * np.sin(y * 0.05 + i * 0.2))
            x_threshold = max(0, min(w, wave_pos + wave_offset))
            
            if x_threshold > 0:
                frame[y, :x_threshold] = img2[y, :x_threshold]
        
        frames.append(frame)
    
    return frames

def pixel_swap_spiral(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Effetto spirale."""
    frames = []
    h, w, c = img1.shape
    center_y, center_x = h // 2, w // 2
    
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
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        if not out.isOpened():
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

def get_video_html(video_path: str) -> str:
    """Genera HTML per preview video."""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    video_b64 = base64.b64encode(video_bytes).decode()
    
    html = f"""
    <video width="100%" height="auto" controls autoplay loop>
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    return html

# --- UI Migliorata ---

def main():
    st.title("🎞️ Frame-to-Frame FX Multi-Image")
    st.markdown("### Genera video con transizioni animate tra 2-5 immagini")
    
    # Sidebar per controlli
    with st.sidebar:
        st.header("⚙️ Impostazioni")
        
        # Selezione risoluzione
        st.subheader("📐 Risoluzione")
        aspect_ratio_name = st.selectbox(
            "Formato video",
            list(ASPECT_RATIOS.keys()),
            help="Seleziona il formato del video"
        )
        
        quality_name = st.selectbox(
            "Qualità video",
            list(QUALITY_PRESETS.keys()),
            index=1,  # Default a 720p
            help="Seleziona la qualità del video"
        )
        
        # Calcola risoluzione finale
        aspect_ratio = ASPECT_RATIOS[aspect_ratio_name]
        max_height = QUALITY_PRESETS[quality_name]
        
        if aspect_ratio:
            target_size = calculate_resolution(aspect_ratio, max_height)
            st.info(f"🎯 Risoluzione: {target_size[0]}x{target_size[1]}")
        else:
            st.info("🎯 Risoluzione: Auto (basata su immagini)")
        
        st.divider()
        
        # Selezione effetto
        effect = st.selectbox(
            "🎨 Effetto di transizione",
            [
                "Dissolvenza (fade)",
                "Morphing",
                "Glitch Lines",
                "Digital Noise",
                "Zoom",
                "Pixel casuali",
                "Blocchi",
                "Onda",
                "Spirale"
            ],
            help="Seleziona l'effetto per le transizioni"
        )
        
        # Parametri video
        st.subheader("🎬 Parametri Video")
        fps = st.slider("FPS", 5, 60, DEFAULT_FPS, help="Frame per secondo")
        frames_per_transition = st.slider(
            "Frame per transizione", 
            10, 300, 
            DEFAULT_FRAMES_PER_TRANSITION,
            help="Numero di frame per transizione"
        )
        
        # Parametri specifici per effetto
        if effect == "Blocchi":
            block_size = st.slider("Dimensione blocchi", 4, 64, 16, step=4)
        else:
            block_size = 16
        
        if effect == "Zoom":
            zoom_factor = st.slider("Fattore zoom", 1.1, 2.0, 1.3, 0.1)
        else:
            zoom_factor = 1.3
    
    # Upload immagini
    uploaded_files = st.file_uploader(
        "📁 Carica immagini (2-5)",
        accept_multiple_files=True,
        type=SUPPORTED_FORMATS,
        help=f"Formati supportati: {', '.join(SUPPORTED_FORMATS)}"
    )
    
    # Mostra preview
    if uploaded_files:
        st.subheader("🖼️ Anteprima immagini")
        
        if len(uploaded_files) > MAX_IMAGES:
            st.warning(f"Verranno utilizzate solo le prime {MAX_IMAGES} immagini")
        
        if len(uploaded_files) < 2:
            st.error("Carica almeno 2 immagini per generare il video!")
            return
        
        # Calcola risoluzione se auto
        if aspect_ratio is None:
            # Calcola basandosi sulle immagini
            preview_images = []
            for file in uploaded_files[:MAX_IMAGES]:
                try:
                    img = Image.open(file)
                    preview_images.append(img)
                except:
                    continue
            
            if preview_images:
                avg_width = sum(img.size[0] for img in preview_images) / len(preview_images)
                avg_height = sum(img.size[1] for img in preview_images) / len(preview_images)
                
                # Scala per rispettare la qualità
                scale = max_height / max(avg_width, avg_height)
                target_size = (int(avg_width * scale), int(avg_height * scale))
                target_size = (target_size[0] - target_size[0] % 2, target_size[1] - target_size[1] % 2)
                
                st.info(f"🎯 Risoluzione auto: {target_size[0]}x{target_size[1]}")
        
        # Mostra grid di anteprime
        cols = st.columns(min(len(uploaded_files), 5))
        
        for i, file in enumerate(uploaded_files[:MAX_IMAGES]):
            try:
                img = Image.open(file)
                with cols[i]:
                    st.image(img, caption=f"Immagine {i+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Errore nell'immagine {i+1}: {e}")
        
        # Stima durata
        if len(uploaded_files) >= 2:
            duration = len(uploaded_files) * frames_per_transition / fps
            st.info(f"⏱️ Durata stimata: {duration:.1f} secondi")
    
    # Genera video
    if st.button("🎬 Genera Video", type="primary"):
        if not uploaded_files or len(uploaded_files) < 2:
            st.error("Carica almeno 2 immagini!")
            return
        
        with st.spinner("🔄 Elaborazione in corso..."):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📥 Processamento immagini...")
                progress_bar.progress(0.1)
                
                # Preprocessa immagini
                processed_images = preprocess_images(uploaded_files, target_size)
                
                if len(processed_images) < 2:
                    st.error("Non sono state processate abbastanza immagini valide!")
                    return
                
                progress_bar.progress(0.3)
                status_text.text("🎨 Generazione effetti...")
                
                # Seleziona funzione effetto
                effect_functions = {
                    "Dissolvenza (fade)": fade_effect,
                    "Morphing": morphing_effect,
                    "Glitch Lines": glitch_lines_effect,
                    "Digital Noise": digital_noise_effect,
                    "Zoom": lambda img1, img2, num_frames: zoom_effect(img1, img2, num_frames, zoom_factor),
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
                    
                    progress = 0.3 + (0.5 * (i + 1) / n_imgs)
                    progress_bar.progress(progress)
                    status_text.text(f"🎬 Transizione {i+1}/{n_imgs}")
                
                progress_bar.progress(0.8)
                status_text.text("💾 Creazione video...")
                
                # Crea video temporaneo
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                success = create_video(all_frames, tmp_path, fps)
                
                if success and os.path.exists(tmp_path):
                    progress_bar.progress(1.0)
                    status_text.text("✅ Video completato!")
                    
                    # Mostra statistiche
                    file_size = os.path.getsize(tmp_path)
                    duration = len(all_frames) / fps
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Durata", f"{duration:.1f}s")
                    with col2:
                        st.metric("📏 Risoluzione", f"{target_size[0]}x{target_size[1]}")
                    with col3:
                        st.metric("🎞️ Frame totali", len(all_frames))
                    with col4:
                        st.metric("📦 Dimensione", f"{file_size/1024/1024:.1f} MB")
                    
                    # Preview del video
                    st.subheader("🎥 Preview Video")
                    
                    try:
                        video_html = get_video_html(tmp_path)
                        st.components.v1.html(video_html, height=400)
                    except Exception as e:
                        st.warning("Preview non disponibile, ma il video è stato generato correttamente")
                        logger.error(f"Errore preview: {e}")
                    
                    # Download
                    with open(tmp_path, "rb") as f:
                        st.download_button(
                            "📥 Scarica Video MP4",
                            f.read(),
                    file_name=f"frame_transition_{effect.lower().replace(' ', '_')}.mp4",
                    mime="video/mp4"
                )
                
                # Cleanup
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Garbage collection
                del all_frames
                del processed_images
                gc.collect()
                
            else:
                st.error("❌ Errore nella creazione del video")
                
        except Exception as e:
            st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
            logger.error(f"Errore main: {e}")
        
        finally:
            # Cleanup finale
            gc.collect()

# Footer
st.divider()
st.markdown("---")
st.markdown("**🎬 Frame-to-Frame FX** - Genera video con transizioni animate tra immagini")

if __name__ == "__main__":
    main()
