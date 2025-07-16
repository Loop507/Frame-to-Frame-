import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import traceback
import math

st.set_page_config(page_title="🎞️ Morphing Studio", layout="wide")
st.title("🔄 Morphing Studio - Effetti Avanzati")

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

def linear_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Morphing lineare semplice."""
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        morphed = (1 - alpha) * img1 + alpha * img2
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    return frames

def wave_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Morphing con effetto onda."""
    h, w = img1.shape[:2]
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Crea griglia di coordinate
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Effetto onda
        wave_intensity = 20 * np.sin(alpha * np.pi)
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

def spiral_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
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
        
        # Effetto spirale
        spiral_factor = alpha * 2 * np.pi
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

def zoom_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Morphing con effetto zoom."""
    h, w = img1.shape[:2]
    frames = []
    center_x, center_y = w // 2, h // 2
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Effetto zoom
        zoom_factor = 1 + 0.5 * np.sin(alpha * np.pi)
        
        # Crea matrice di trasformazione
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
        
        # Applica trasformazione
        img1_transformed = cv2.warpAffine(img1, M, (w, h))
        img2_transformed = cv2.warpAffine(img2, M, (w, h))
        
        # Morphing lineare
        morphed = (1 - alpha) * img1_transformed + alpha * img2_transformed
        frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
    
    return frames

def generate_morph_with_progress(img1: np.ndarray, img2: np.ndarray, num_frames: int, effect: str):
    """Genera morphing con barra di progresso."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if effect == "Linear":
            frames = linear_morph(img1, img2, num_frames)
        elif effect == "Wave":
            frames = wave_morph(img1, img2, num_frames)
        elif effect == "Spiral":
            frames = spiral_morph(img1, img2, num_frames)
        elif effect == "Zoom":
            frames = zoom_morph(img1, img2, num_frames)
        else:
            frames = linear_morph(img1, img2, num_frames)
        
        # Simula progresso per effetti che non hanno loop interno
        for i in range(num_frames):
            progress = (i + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"Generazione frame {i + 1}/{num_frames} - Effetto: {effect}")
        
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
    
    # Effect selector
    st.sidebar.header("🎨 Effetti")
    effect = st.sidebar.selectbox("Tipo di Morphing", ["Linear", "Wave", "Spiral", "Zoom"])
    
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
        
        # Display images in grid
        st.header("🖼️ Anteprima Immagini")
        cols = st.columns(min(4, num_images))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(img, caption=f"Immagine {i+1}", use_column_width=True)
        
        # Generate morphing
        if st.button("🚀 Genera Morphing", type="primary", use_container_width=True):
            
            # Calculate frames per transition
            transitions = num_images - 1
            frames_per_transition = num_frames // transitions
            
            st.info(f"🎬 Generando {transitions} transizioni con {frames_per_transition} frame ciascuna")
            
            all_frames = []
            
            # Generate morphing between consecutive images
            for i in range(transitions):
                st.subheader(f"🔄 Transizione {i+1}/{transitions}: Immagine {i+1} → Immagine {i+2}")
                
                frames = generate_morph_with_progress(
                    images[i], 
                    images[i+1], 
                    frames_per_transition, 
                    effect
                )
                
                if not frames:
                    st.error(f"❌ Errore nella transizione {i+1}")
                    return
                
                all_frames.extend(frames)
            
            st.success(f"✅ Morphing completato! {len(all_frames)} frame totali generati")
            
            # Save and display result
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(suffix=f".{export_format.lower()}", delete=False) as tmp:
                    temp_file = tmp.name
                    
                    with st.spinner(f"💾 Salvataggio {export_format}..."):
                        if export_format == "GIF":
                            success = save_as_gif(all_frames, temp_file, fps)
                            file_label = f"morphing_{num_images}images_{duration_seconds}s.gif"
                            mime_type = "image/gif"
                        else:
                            success = save_as_mp4(all_frames, temp_file, fps)
                            file_label = f"morphing_{num_images}images_{duration_seconds}s.mp4"
                            mime_type = "video/mp4"
                    
                    if not success:
                        st.error("❌ Errore nel salvataggio")
                        return
                    
                    # Display preview
                    st.header("🎬 Anteprima Risultato")
                    if export_format == "GIF":
                        st.image(temp_file, caption=f"Morphing GIF - {effect} Effect", use_column_width=True)
                    else:
                        st.video(temp_file)
                    
                    # Download button
                    with open(temp_file, "rb") as f:
                        st.download_button(
                            f"📥 Scarica {export_format} ({duration_seconds}s)",
                            f.read(),
                            file_name=file_label,
                            mime=mime_type,
                            type="primary",
                            use_container_width=True
                        )
                    
                    # File info
                    file_size = os.path.getsize(temp_file)
                    st.info(f"ℹ️ File: {file_label} | Dimensioni: {file_size/1024/1024:.1f} MB | {fps} FPS | {len(all_frames)} frame | Effetto: {effect}")
                    
            except Exception as e:
                st.error(f"❌ Errore imprevisto: {str(e)}")
                st.code(traceback.format_exc())
            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
    else:
        st.info("👆 Carica almeno 2 immagini per iniziare")
        
        # Show example
        st.header("🎯 Funzionalità")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📐 Aspect Ratios**")
            st.write("• 1:1 Square")
            st.write("• 16:9 Widescreen")
            st.write("• 9:16 Portrait")
            st.write("• Custom")
        
        with col2:
            st.write("**🎨 Effetti**")
            st.write("• Linear Morphing")
            st.write("• Wave Effect")
            st.write("• Spiral Effect")
            st.write("• Zoom Effect")
        
        with col3:
            st.write("**⚡ Features**")
            st.write("• Fino a 10 immagini")
            st.write("• Durata personalizzabile")
            st.write("• Export MP4/GIF")
            st.write("• Barra di progresso")

if __name__ == "__main__":
    main()
