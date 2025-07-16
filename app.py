import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import traceback

st.set_page_config(page_title="🎞️ Morphing Base", layout="wide")
st.title("🔄 Morphing tra Immagini (GIF / MP4)")

MAX_IMAGES = 2
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
DEFAULT_FPS = 24
DEFAULT_FRAMES = 30
MAX_RESOLUTION = 1920 * 1080  # Limite per evitare problemi di memoria

def load_image(uploaded_file) -> Optional[np.ndarray]:
    """Carica e converte un'immagine in array numpy con gestione errori."""
    try:
        img = Image.open(uploaded_file).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dell'immagine '{uploaded_file.name}': {str(e)}")
        return None

def validate_image_size(width: int, height: int) -> bool:
    """Valida le dimensioni dell'immagine per evitare problemi di memoria."""
    total_pixels = width * height
    if total_pixels > MAX_RESOLUTION:
        st.error(f"❌ Dimensioni troppo grandi! Massimo consentito: {MAX_RESOLUTION:,} pixel")
        return False
    return True

def resize_to_target(img: np.ndarray, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Ridimensiona l'immagine mantenendo le proporzioni con crop centrale."""
    try:
        original = Image.fromarray(img)
        target_ratio = target_size[0] / target_size[1]
        original_ratio = original.width / original.height

        if original_ratio > target_ratio:
            # Immagine più larga, crop orizzontalmente
            new_width = int(original.height * target_ratio)
            left = (original.width - new_width) // 2
            cropped = original.crop((left, 0, left + new_width, original.height))
        else:
            # Immagine più alta, crop verticalmente
            new_height = int(original.width / target_ratio)
            top = (original.height - new_height) // 2
            cropped = original.crop((0, top, original.width, top + new_height))

        return np.array(cropped.resize(target_size, Image.LANCZOS))
    except Exception as e:
        st.error(f"❌ Errore nel ridimensionamento: {str(e)}")
        return None

def grid_warp_morph(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Crea morphing tra due immagini con effetto di distorsione a griglia."""
    try:
        h, w = img1.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        frames = []

        # Progress bar per il morphing
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
            # Calcola distorsione sinusoidale
            distortion = 15 * alpha * (1 - alpha)
            dx = distortion * np.sin(y / h * 2 * np.pi + alpha * np.pi)
            dy = distortion * np.cos(x / w * 2 * np.pi + alpha * np.pi)
            
            # Applica distorsione con clipping
            xn = np.clip(x + dx, 0, w - 1)
            yn = np.clip(y + dy, 0, h - 1)
            
            # Interpolazione bilineare
            xf = np.floor(xn).astype(int)
            yf = np.floor(yn).astype(int)
            xc = np.clip(xf + 1, 0, w - 1)
            yc = np.clip(yf + 1, 0, h - 1)
            
            a = xn - xf
            b = yn - yf
            
            morphed = np.empty_like(img1)

            # Morphing per canale colore
            for c in range(3):
                chan = img1[..., c] * (1 - alpha) + img2[..., c] * alpha
                top = (1 - a) * chan[yf, xf] + a * chan[yf, xc]
                bottom = (1 - a) * chan[yc, xf] + a * chan[yc, xc]
                morphed[..., c] = (1 - b) * top + b * bottom

            frames.append(np.clip(morphed, 0, 255).astype(np.uint8))
            
            # Aggiorna progress bar
            progress = (i + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"Generazione frame {i + 1}/{num_frames}")

        progress_bar.empty()
        status_text.empty()
        return frames
        
    except Exception as e:
        st.error(f"❌ Errore durante il morphing: {str(e)}")
        return []

def save_as_gif(frames: List[np.ndarray], path: str, fps: int) -> bool:
    """Salva i frame come GIF animata."""
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
    """Salva i frame come video MP4."""
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
    st.sidebar.header("⚙️ Impostazioni")
    
    # Controlli sidebar con validazione
    target_width = st.sidebar.number_input("Larghezza", 100, 1920, 800)
    target_height = st.sidebar.number_input("Altezza", 100, 1080, 600)
    
    # Valida dimensioni
    if not validate_image_size(target_width, target_height):
        st.stop()
    
    fps = st.sidebar.number_input("FPS", 5, 60, DEFAULT_FPS)
    num_frames = st.sidebar.number_input("Frame", 10, 300, DEFAULT_FRAMES)
    
    # Avviso per operazioni pesanti
    if num_frames > 100:
        st.sidebar.warning("⚠️ Molti frame potrebbero richiedere tempo")
    
    # Upload files
    uploaded_files = st.file_uploader(
        "Carica esattamente 2 immagini",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help="Formati supportati: JPG, JPEG, PNG"
    )

    export_format = st.selectbox("Formato esportazione", ["GIF", "MP4"])

    # Validazione numero file
    if uploaded_files:
        if len(uploaded_files) != 2:
            st.warning(f"⚠️ Hai caricato {len(uploaded_files)} immagini. Carica esattamente 2 immagini.")
            return
        
        target_size = (target_width, target_height)
        
        # Carica e processa immagini
        with st.spinner("📥 Caricamento immagini..."):
            img1 = load_image(uploaded_files[0])
            img2 = load_image(uploaded_files[1])
            
            if img1 is None or img2 is None:
                st.error("❌ Impossibile caricare le immagini. Riprova.")
                return
            
            # Ridimensiona immagini
            img1_resized = resize_to_target(img1, target_size)
            img2_resized = resize_to_target(img2, target_size)
            
            if img1_resized is None or img2_resized is None:
                st.error("❌ Impossibile ridimensionare le immagini. Riprova.")
                return

        # Mostra anteprima immagini
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1_resized, caption=f"Immagine 1: {uploaded_files[0].name}", use_column_width=True)
        with col2:
            st.image(img2_resized, caption=f"Immagine 2: {uploaded_files[1].name}", use_column_width=True)

        # Genera morphing
        if st.button("🚀 Genera Morphing", type="primary"):
            with st.spinner("🌀 Generazione morphing in corso..."):
                frames = grid_warp_morph(img1_resized, img2_resized, num_frames)
                
                if not frames:
                    st.error("❌ Errore nella generazione del morphing")
                    return
                
                st.success(f"✅ Morphing completato! {len(frames)} frame generati")

            # Salva e mostra risultato
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(suffix=f".{export_format.lower()}", delete=False) as tmp:
                    temp_file = tmp.name
                    
                    with st.spinner(f"💾 Salvataggio {export_format}..."):
                        if export_format == "GIF":
                            success = save_as_gif(frames, temp_file, fps)
                            file_label = "morphing.gif"
                            mime_type = "image/gif"
                        else:
                            success = save_as_mp4(frames, temp_file, fps)
                            file_label = "morphing.mp4"
                            mime_type = "video/mp4"
                    
                    if not success:
                        st.error("❌ Errore nel salvataggio del file")
                        return
                    
                    # Mostra anteprima
                    st.subheader("🎬 Anteprima risultato")
                    if export_format == "GIF":
                        st.image(temp_file, caption="Morphing GIF", use_column_width=True)
                    else:
                        st.video(temp_file)
                    
                    # Download button
                    with open(temp_file, "rb") as f:
                        st.download_button(
                            f"📥 Scarica {export_format}",
                            f.read(),
                            file_name=file_label,
                            mime=mime_type,
                            type="primary"
                        )
                    
                    st.info(f"ℹ️ File generato: {file_label} ({fps} FPS, {num_frames} frame)")
                    
            except Exception as e:
                st.error(f"❌ Errore imprevisto: {str(e)}")
                st.code(traceback.format_exc())
            finally:
                # Cleanup file temporaneo
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
    else:
        st.info("👆 Carica 2 immagini per iniziare")

if __name__ == "__main__":
    main()
