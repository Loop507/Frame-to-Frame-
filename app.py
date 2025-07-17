import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import traceback
import math
import random

st.set_page_config(page_title="🎞️ Frame to Frame", layout="wide")

# Titolo personalizzato con HTML
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #1f77b4; font-size: 3rem; margin-bottom: 5px;">🎞️ Frame to Frame</h1>
        <p style="color: #666; font-size: 1.2rem; margin-top: 0;">by Loop507</p>
    </div>
""", unsafe_allow_html=True)

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
DEFAULT_FPS = 30
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "16:9 (Widescreen)": (16, 9),
    "9:16 (Portrait)": (9, 16),
    "4:3 (Classic)": (4, 3),
    "3:4 (Portrait Classic)": (3, 4),
    "21:9 (Ultrawide)": (21, 9),
    "Custom": None
}

# Intensità effetti
EFFECT_INTENSITIES = {
    "Soft": 0.3,
    "Medium": 0.6,
    "Hard": 1.0
}

def load_image(uploaded_file) -> Optional[np.ndarray]:
    """Carica e converte un'immagine in array numpy."""
    try:
        img = Image.open(uploaded_file).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dell'immagine '{uploaded_file.name}': {str(e)}")
        return None

def calculate_size_from_ratio(ratio: Tuple[int, int], base_size: int = 800) -> Tuple[int, int]:
    """Calcola dimensioni mantenendo l'aspect ratio."""
    w_ratio, h_ratio = ratio
    if w_ratio >= h_ratio:
        width = base_size
        height = int(base_size * h_ratio / w_ratio)
    else:
        height = base_size
        width = int(base_size * w_ratio / h_ratio)
    return width, height

def resize_to_target(img: np.ndarray, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Ridimensiona l'immagine con crop intelligente."""
    try:
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
    except Exception as e:
        st.error(f"❌ Errore nel ridimensionamento: {str(e)}")
        return None

def linear_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing lineare semplice."""
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        # Applica intensità
        smooth_alpha = alpha * intensity + (1 - intensity) * 0.5
        morphed = (1 - smooth_alpha) * img1 + smooth_alpha * img2
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def wave_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto onda."""
    h, w = img1.shape[:2]
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Crea griglia di coordinate
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Effetto onda con intensità variabile
        wave_intensity = 20 * intensity * np.sin(alpha * np.pi)
        dx = wave_intensity * np.sin(2 * np.pi * y / h + alpha * 4 * np.pi)
        dy = wave_intensity * np.cos(2 * np.pi * x / w + alpha * 4 * np.pi)
        
        # Applica distorsione
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        
        # Interpolazione
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def spiral_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto spirale."""
    h, w = img1.shape[:2]
    frames = []
    center_x, center_y = w // 2, h // 2
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Crea griglia di coordinate
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Converti in coordinate polari
        dx = x - center_x
        dy = y - center_y
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Effetto spirale con intensità
        spiral_factor = alpha * 2 * np.pi * intensity
        angle_new = angle + spiral_factor * (radius / max(w, h))
        
        # Riconverti in coordinate cartesiane
        x_new = center_x + radius * np.cos(angle_new)
        y_new = center_y + radius * np.sin(angle_new)
        
        x_new = np.clip(x_new, 0, w - 1)
        y_new = np.clip(y_new, 0, h - 1)
        
        # Interpolazione
        morphed = np.empty_like(img1)
        for c in range(3):
            chan = (1 - alpha) * img1[..., c] + alpha * img2[..., c]
            morphed[..., c] = cv2.remap(chan, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def zoom_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto zoom."""
    h, w = img1.shape[:2]
    frames = []
    center_x, center_y = w // 2, h // 2
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Effetto zoom con intensità
        zoom_factor = 1 + (0.5 * intensity) * np.sin(alpha * np.pi)
        
        # Crea matrice di trasformazione
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
        
        # Applica trasformazione
        img1_transformed = cv2.warpAffine(img1, M, (w, h))
        img2_transformed = cv2.warpAffine(img2, M, (w, h))
        
        # Morphing lineare
        morphed = (1 - alpha) * img1_transformed + alpha * img2_transformed
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def glitch_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto glitch digitale."""
    h, w = img1.shape[:2]
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Base morphing
        base_morph = (1 - alpha) * img1 + alpha * img2
        
        # Effetto glitch con intensità
        glitch_intensity = 0.3 * intensity * np.sin(alpha * np.pi * 4)
        
        if abs(glitch_intensity) > 0.1:
            # Shift di canali RGB
            result = base_morph.copy()
            shift = int(glitch_intensity * 20 * intensity)
            
            # Shift canale rosso
            if shift > 0:
                result[:, shift:, 0] = base_morph[:, :-shift, 0]
            else:
                result[:, :shift, 0] = base_morph[:, -shift:, 0]
            
            # Shift canale blu (direzione opposta)
            if shift > 0:
                result[:, :-shift, 2] = base_morph[:, shift:, 2]
            else:
                result[:, -shift:, 2] = base_morph[:, :shift, 2]
            
            # Linee orizzontali random
            num_lines = int(random.randint(1, 5) * intensity)
            for _ in range(num_lines):
                y = random.randint(0, h - 1)
                thickness = random.randint(1, 3)
                end_y = min(y + thickness, h)
                
                # Shift orizzontale della linea
                line_shift = int(random.randint(-30, 30) * intensity)
                if line_shift > 0:
                    result[y:end_y, line_shift:] = base_morph[y:end_y, :-line_shift]
                elif line_shift < 0:
                    result[y:end_y, :line_shift] = base_morph[y:end_y, -line_shift:]
            
            frames.append(np.clip(result, 0, 255).astype(np.uint8))
        else:
            frames.append(np.clip(base_morph, 0, 255).astype(np.uint8))
    
    return frames

def swap_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto scambio a blocchi."""
    h, w = img1.shape[:2]
    frames = []
    
    # Dimensione dei blocchi basata su intensità
    block_size = min(w, h) // int(8 * intensity + 2)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        result = img1.copy()
        
        # Numero di blocchi da scambiare basato su alpha e intensità
        num_blocks = int(alpha * intensity * (h // block_size) * (w // block_size))
        
        blocks_swapped = 0
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if blocks_swapped >= num_blocks:
                    break
                
                # Determina se scambiare questo blocco
                if random.random() < alpha * intensity:
                    y_end = min(y + block_size, h)
                    x_end = min(x + block_size, w)
                    
                    # Scambia con img2
                    result[y:y_end, x:x_end] = img2[y:y_end, x:x_end]
                    blocks_swapped += 1
            
            if blocks_swapped >= num_blocks:
                break
        
        frames.append(result)
    
    return frames

def pixel_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto pixelato."""
    h, w = img1.shape[:2]
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Calcola il livello di pixelizzazione con intensità
        pixel_level = int(2 + 30 * intensity * np.sin(alpha * np.pi))
        
        # Base morphing
        base_morph = (1 - alpha) * img1 + alpha * img2
        
        # Effetto pixel
        small_h, small_w = h // pixel_level, w // pixel_level
        
        # Ridimensiona a risoluzione molto bassa
        small_img = cv2.resize(base_morph, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        # Ridimensiona di nuovo alle dimensioni originali
        pixelated = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
        
        frames.append(np.clip(pixelated, 0, 255).astype(np.uint8))
    
    return frames

def distorted_lines_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con linee distorte."""
    h, w = img1.shape[:2]
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Base morphing
        base_morph = (1 - alpha) * img1 + alpha * img2
        result = base_morph.copy()
        
        # Effetto linee distorte con intensità
        distortion_intensity = 0.5 * intensity * np.sin(alpha * np.pi * 3)
        
        if abs(distortion_intensity) > 0.1:
            # Linee orizzontali distorte
            line_spacing = int(10 / intensity) if intensity > 0 else 10
            for y in range(0, h, line_spacing):
                if y < h:
                    # Crea distorsione sinusoidale
                    distortion = distortion_intensity * 30 * np.sin(np.arange(w) * 2 * np.pi / w * 5)
                    
                    for x in range(w):
                        source_x = int(x + distortion[x])
                        source_x = max(0, min(w - 1, source_x))
                        
                        if y < h:
                            result[y, x] = base_morph[y, source_x]
        
        frames.append(np.clip(result, 0, 255).astype(np.uint8))
    
    return frames

def slice_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto slice (fette)."""
    h, w = img1.shape[:2]
    frames = []
    
    slice_width = w // int(20 * intensity)  # Più slice con intensità alta
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        result = img1.copy()
        
        # Numero di slice da sostituire
        num_slices = int(alpha * intensity * (w // slice_width))
        
        for slice_idx in range(num_slices):
            x_start = slice_idx * slice_width
            x_end = min(x_start + slice_width, w)
            
            # Offset verticale per effetto dinamico
            offset = int(20 * intensity * np.sin(alpha * np.pi * 2 + slice_idx * 0.5))
            
            # Applica slice con offset
            for y in range(h):
                source_y = (y + offset) % h
                result[y, x_start:x_end] = img2[source_y, x_start:x_end]
        
        frames.append(result)
    
    return frames

def rotation_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto rotazione."""
    h, w = img1.shape[:2]
    frames = []
    center_x, center_y = w // 2, h // 2
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Angolo di rotazione con intensità
        angle = alpha * 360 * 2 * intensity  # Rotazioni variabili
        
        # Matrice di rotazione
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # Ruota entrambe le immagini
        img1_rotated = cv2.warpAffine(img1, M, (w, h))
        img2_rotated = cv2.warpAffine(img2, M, (w, h))
        
        # Morphing lineare
        morphed = (1 - alpha) * img1_rotated + alpha * img2_rotated
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def ripple_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto ripple (increspatura)."""
    h, w = img1.shape[:2]
    frames = []
    center_x, center_y = w // 2, h // 2
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Base morphing
        base_morph = (1 - alpha) * img1 + alpha * img2
        
        # Effetto ripple con intensità
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Distanza dal centro
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Parametri ripple
        ripple_amplitude = 15 * intensity * np.sin(alpha * np.pi)
        ripple_frequency = 0.1 * intensity
        ripple_speed = alpha * 10 * intensity
        
        # Calcola distorsione
        distortion = ripple_amplitude * np.sin(distance * ripple_frequency + ripple_speed)
        
        # Applica distorsione radiale
        x_new = center_x + (dx + distortion * dx / (distance + 1))
        y_new = center_y + (dy + distortion * dy / (distance + 1))
        
        # Clamp coordinates
        x_new = np.clip(x_new, 0, w - 1)
        y_new = np.clip(y_new, 0, h - 1)
        
        # Applica distorsione
        result = np.empty_like(base_morph)
        for c in range(3):
            result[..., c] = cv2.remap(base_morph[..., c], x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        
        frames.append(np.clip(result, 0, 255).astype(np.uint8))
    
    return frames

def random_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int, intensity: float = 1.0) -> List[np.ndarray]:
    """Morphing con effetto casuale che cambia dinamicamente."""
    # Lista di tutti gli effetti disponibili
    effects = [
        wave_morph, spiral_morph, zoom_morph, glitch_morph, 
        swap_morph, pixel_morph, distorted_lines_morph, 
        slice_morph, rotation_morph, ripple_morph
    ]
    
    # Divide i frame in segmenti casuali
    segments = random.randint(3, 6)  # 3-6 segmenti diversi
    frames_per_segment = num_frames // segments
    
    all_frames = []
    
    for segment in range(segments):
        start_frame = segment * frames_per_segment
        end_frame = (segment + 1) * frames_per_segment if segment < segments - 1 else num_frames
        segment_frames = end_frame - start_frame
        
        # Scegli un effetto casuale per questo segmento
        effect_function = random.choice(effects)
        
        # Genera i frame per questo segmento
        segment_result = effect_function(img1, img2, segment_frames, intensity)
        all_frames.extend(segment_result)
    
    return all_frames

def generate_morph_with_progress(img1: np.ndarray, img2: np.ndarray, num_frames: int, effect: str, intensity: float = 1.0):
    """Genera morphing con barra di progresso."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Seleziona l'effetto
        effect_functions = {
            "Linear": linear_morph,
            "Wave": wave_morph,
            "Spiral": spiral_morph,
            "Zoom": zoom_morph,
            "Glitch": glitch_morph,
            "Swap": swap_morph,
            "Pixel": pixel_morph,
            "Distorted Lines": distorted_lines_morph,
            "Slice": slice_morph,
            "Rotation": rotation_morph,
            "Ripple": ripple_morph,
            "Random": random_morph
        }
        
        morph_function = effect_functions.get(effect, linear_morph)
        frames = morph_function(img1, img2, num_frames, intensity)
        
        # Simula progresso
        for i in range(num_frames):
            progress = (i + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"Generazione frame {i + 1}/{num_frames} - Effetto: {effect} ({intensity:.1f})")
        
        progress_bar.empty()
        status_text.empty()
        return frames
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Errore durante il morphing: {str(e)}")
        return []

def save_as_gif(frames: List[np.ndarray], path: str, fps: int) -> bool:
    """Salva i frame come GIF."""
    try:
        pil_frames = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        return True
    except Exception as e:
        st.error(f"❌ Errore nel salvataggio GIF: {str(e)}")
        return False

def save_as_mp4(frames: List[np.ndarray], path: str, fps: int) -> bool:
    """Salva i frame come MP4."""
    try:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return True
    except Exception as e:
        st.error(f"❌ Errore nel salvataggio MP4: {str(e)}")
        return False

def main():
    # Sidebar settings
    st.sidebar.header("⚙️ Impostazioni Video")
    
    # Aspect ratio selector
    aspect_choice = st.sidebar.selectbox("📐 Aspect Ratio", list(ASPECT_RATIOS.keys()))
    
    if aspect_choice == "Custom":
        target_width = st.sidebar.number_input("Larghezza", 100, 1920, 800)
        target_height = st.sidebar.number_input("Altezza", 100, 1080, 600)
    else:
        ratio = ASPECT_RATIOS[aspect_choice]
        base_size = st.sidebar.slider("📏 Dimensione Base", 400, 1200, 800, 50)
        target_width, target_height = calculate_size_from_ratio(ratio, base_size)
        st.sidebar.info(f"Risoluzione: {target_width}x{target_height}")
    
    # Duration settings
    st.sidebar.header("⏱️ Durata")
    duration_seconds = st.sidebar.slider("Durata (secondi)", 1, 30, 5)
    fps = st.sidebar.number_input("FPS", 10, 60, DEFAULT_FPS)
    num_frames = int(duration_seconds * fps)
    st.sidebar.info(f"Total frames: {num_frames}")
    
    # Effect selector with new effects
    st.sidebar.header("🎨 Effetti")
    effect = st.sidebar.selectbox("Tipo di Morphing", [
        "Linear", "Wave", "Spiral", "Zoom", 
        "Glitch", "Swap", "Pixel", "Distorted Lines", 
        "Slice", "Rotation", "Ripple", "Random"
    ])
    
    # Intensity control
    st.sidebar.header("🔥 Intensità Effetto")
    intensity_level = st.sidebar.selectbox("Livello di Intensità", ["Soft", "Medium", "Hard"])
    intensity = EFFECT_INTENSITIES[intensity_level]
    
    # Show intensity info
    intensity_info = {
        "Soft": "🌸 Effetto delicato e sottile",
        "Medium": "⚡ Effetto bilanciato",
        "Hard": "🔥 Effetto intenso e drammatico"
    }
    st.sidebar.info(intensity_info[intensity_level])
    
    # Export format
    export_format = st.sidebar.selectbox("💾 Formato", ["MP4", "GIF"])
    
    # Multiple file uploader
    st.header("📸 Carica Immagini")
    uploaded_files = st.file_uploader(
        "Carica almeno 2 immagini (massimo 10)",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help="Trascina qui le tue immagini o clicca per selezionare"
    )
    
    if uploaded_files:
        num_images = len(uploaded_files)
        
        if num_images < 2:
            st.warning("⚠️ Carica almeno 2 immagini per creare il morphing")
            return
        elif num_images > 10:
            st.error("❌ Massimo 10 immagini consentite")
            return
        
        st.success(f"✅ {num_images} immagini caricate")
        
        # Load and resize images
        with st.spinner("📥 Elaborazione immagini..."):
            images = []
            target_size = (target_width, target_height)
            
            for i, uploaded_file in enumerate(uploaded_files):
                img = load_image(uploaded_file)
                if img is not None:
                    img_resized = resize_to_target(img, target_size)
                    if img_resized is not None:
                        images.append(img_resized)
                        
            if len(images) != num_images:
                st.error("❌ Errore nel caricamento di alcune immagini")
                return
        
        # Generate morphing
        if st.button("🚀 Genera Morphing", type="primary", use_container_width=True):
            
            # Calculate frames per transition
            transitions = num_images - 1
            frames_per_transition = num_frames // transitions
            
            st.info(f"🎬 Generando {transitions} transizioni con {frames_per_transition} frame ciascuna - Intensità: {intensity_level}")
            
            all_frames = []
            
            # Generate morphing between consecutive images
            for i in range(transitions):
                st.subheader(f"🔄 Transizione {i+1}/{transitions}: Immagine {i+1} → Immagine {i+2}")
                
                frames = generate_morph_with_progress(
                    images[i], 
                    images[i+1], 
                    frames_per_transition, 
                    effect,
                    intensity
                )
                
                if not frames:
                    st.error(f"❌ Errore nella transizione {i+1}")
return
                
                all_frames.extend(frames)
            
            if all_frames:
                st.success(f"✅ {len(all_frames)} frame generati con successo!")
                
                # Show preview
                st.subheader("🎬 Preview")
                preview_container = st.container()
                
                with preview_container:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Show first, middle, and last frames
                        preview_frames = [
                            all_frames[0],
                            all_frames[len(all_frames)//2],
                            all_frames[-1]
                        ]
                        
                        for idx, frame in enumerate(preview_frames):
                            frame_labels = ["🎬 Primo Frame", "🎯 Frame Centrale", "🏁 Ultimo Frame"]
                            st.image(frame, caption=frame_labels[idx], use_container_width=True)
                
                # Save and download
                st.subheader("💾 Download")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format.lower()}") as tmp_file:
                    if export_format == "MP4":
                        success = save_as_mp4(all_frames, tmp_file.name, fps)
                    else:
                        success = save_as_gif(all_frames, tmp_file.name, fps)
                    
                    if success:
                        with open(tmp_file.name, "rb") as f:
                            file_data = f.read()
                        
                        file_size = len(file_data) / (1024 * 1024)  # Size in MB
                        st.success(f"✅ File generato: {file_size:.2f} MB")
                        
                        # Download button
                        st.download_button(
                            label=f"📥 Scarica {export_format}",
                            data=file_data,
                            file_name=f"morphing_{effect.lower()}_{intensity_level.lower()}.{export_format.lower()}",
                            mime=f"video/{export_format.lower()}" if export_format == "MP4" else "image/gif",
                            use_container_width=True
                        )
                        
                        # Technical info
                        with st.expander("🔧 Informazioni Tecniche"):
                            st.write(f"**Risoluzione:** {target_width}x{target_height}")
                            st.write(f"**FPS:** {fps}")
                            st.write(f"**Durata:** {duration_seconds} secondi")
                            st.write(f"**Frame totali:** {len(all_frames)}")
                            st.write(f"**Effetto:** {effect}")
                            st.write(f"**Intensità:** {intensity_level} ({intensity})")
                            st.write(f"**Formato:** {export_format}")
                            st.write(f"**Dimensione file:** {file_size:.2f} MB")
                            st.write(f"**Transizioni:** {transitions}")
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
            else:
                st.error("❌ Nessun frame generato")
        
        # Show loaded images preview
        st.subheader("🖼️ Anteprima Immagini")
        cols = st.columns(min(len(images), 5))
        for i, img in enumerate(images):
            with cols[i % 5]:
                st.image(img, caption=f"Immagine {i+1}", use_container_width=True)
            
            # Show next row if more than 5 images
            if (i + 1) % 5 == 0 and i + 1 < len(images):
                cols = st.columns(min(len(images) - i - 1, 5))
    
    else:
        # Show demo/help section
        st.header("🎯 Come Funziona")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📝 Istruzioni
            1. **Carica le immagini**: Seleziona 2-10 immagini
            2. **Scegli l'aspect ratio**: Personalizza le dimensioni
            3. **Imposta la durata**: Controlla velocità e FPS
            4. **Seleziona l'effetto**: Scegli tra 12 effetti diversi
            5. **Regola l'intensità**: Soft, Medium o Hard
            6. **Genera e scarica**: MP4 o GIF
            """)
        
        with col2:
            st.markdown("""
            ### 🎨 Effetti Disponibili
            - **Linear**: Transizione fluida classica
            - **Wave**: Effetto ondulatorio
            - **Spiral**: Rotazione spirale
            - **Zoom**: Ingrandimento dinamico
            - **Glitch**: Distorsione digitale
            - **Swap**: Scambio a blocchi
            - **Pixel**: Effetto pixelato
            - **Distorted Lines**: Linee distorte
            - **Slice**: Effetto a fette
            - **Rotation**: Rotazione continua
            - **Ripple**: Increspatura radiale
            - **Random**: Mix di effetti casuali
            """)
        
        st.markdown("""
        ### 🎬 Suggerimenti
        - **Immagini simili**: Migliori risultati con soggetti simili
        - **Aspect ratio**: Mantieni coerenza per evitare distorsioni
        - **Intensità**: Inizia con "Medium" per un buon equilibrio
        - **FPS**: 24-30 FPS per fluidità, 15-20 per file più piccoli
        - **Durata**: 3-7 secondi sono ideali per la maggior parte degli usi
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
<p>🎬 <strong>Frame to Frame</strong> - Crea morphing video straordinari</p>
<p>Made with ❤️ by Loop507</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
