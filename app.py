import streamlit as st
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import random
from io import BytesIO

# --- Funzioni di effetto (uguali al tuo codice originale) ---

def load_image(path, size):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(size)
        return np.array(img).astype(np.float32)
    except FileNotFoundError:
        st.error(f"File non trovato: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Errore nel caricamento dell'immagine: {e}")
        st.stop()

def calculate_video_duration(frames, fps, num_transitions=1):
    total_frames = frames * num_transitions
    return total_frames / fps

def fade_effect(img1, img2, num_frames):
    frames = []
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        blended = (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
        frames.append(blended)
    return frames

def zoom_effect(img1, img2, num_frames, zoom_factor=1.2):
    frames = []
    h, w = img1.shape[:2]
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        scale = 1 + (zoom_factor - 1) * alpha
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < h or new_w < w:
            new_h, new_w = h, w
        img1_scaled = np.array(Image.fromarray(img1.astype(np.uint8)).resize((new_w, new_h)))
        img2_scaled = np.array(Image.fromarray(img2.astype(np.uint8)).resize((new_w, new_h)))
        offset_x = (img1_scaled.shape[1] - w) // 2
        offset_y = (img1_scaled.shape[0] - h) // 2
        if offset_x >= 0 and offset_y >= 0:
            frame = (img1_scaled[offset_y:offset_y+h, offset_x:offset_x+w] * (1 - alpha) +
                     img2_scaled[offset_y:offset_y+h, offset_x:offset_x+w] * alpha).astype(np.uint8)
        else:
            frame = (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
        frames.append(frame)
    return frames

def pixel_swap_random(img1, img2, num_frames):
    frames = []
    h, w, c = img1.shape
    total_pixels = h * w
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        num_swapped = int(total_pixels * alpha)
        frame = img1.copy()
        swap_coords = []
        for _ in range(num_swapped):
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)
            swap_coords.append((y, x))
        for y, x in swap_coords:
            frame[y, x] = img2[y, x]
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_swap_blocks(img1, img2, num_frames, block_size=8):
    frames = []
    h, w, c = img1.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        num_swapped_blocks = int(total_blocks * alpha)
        frame = img1.copy()
        block_coords = []
        for _ in range(num_swapped_blocks):
            by = random.randint(0, blocks_h-1)
            bx = random.randint(0, blocks_w-1)
            block_coords.append((by, bx))
        for by, bx in block_coords:
            y_start = by * block_size
            y_end = min(y_start + block_size, h)
            x_start = bx * block_size
            x_end = min(x_start + block_size, w)
            frame[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_swap_wave(img1, img2, num_frames):
    frames = []
    h, w, c = img1.shape
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        frame = img1.copy()
        wave_pos = int(w * alpha)
        for y in range(h):
            wave_offset = int(10 * np.sin(y * 0.1 + i * 0.3))
            x_threshold = wave_pos + wave_offset
            if x_threshold > 0:
                end_x = min(x_threshold, w)
                frame[y, :end_x] = img2[y, :end_x]
        frames.append(frame.astype(np.uint8))
    return frames

def pixel_swap_spiral(img1, img2, num_frames):
    frames = []
    h, w, c = img1.shape
    center_y, center_x = h // 2, w // 2
    spiral_coords = []
    max_radius = int(np.sqrt(center_x**2 + center_y**2))
    for radius in range(max_radius + 1):
        for angle in np.linspace(0, 2*np.pi, max(8, radius*2), endpoint=False):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                spiral_coords.append((y, x))
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        num_swapped = int(len(spiral_coords) * alpha)
        frame = img1.copy()
        for j in range(num_swapped):
            if j < len(spiral_coords):
                y, x = spiral_coords[j]
                frame[y, x] = img2[y, x]
        frames.append(frame.astype(np.uint8))
    return frames

def generate_transition_frames(img1, img2, num_frames, effect_type, block_size=8, zoom_factor=1.2):
    if effect_type == "fade":
        return fade_effect(img1, img2, num_frames)
    elif effect_type == "zoom":
        return zoom_effect(img1, img2, num_frames, zoom_factor)
    elif effect_type == "pixel_random":
        return pixel_swap_random(img1, img2, num_frames)
    elif effect_type == "pixel_blocks":
        return pixel_swap_blocks(img1, img2, num_frames, block_size)
    elif effect_type == "pixel_wave":
        return pixel_swap_wave(img1, img2, num_frames)
    elif effect_type == "pixel_spiral":
        return pixel_swap_spiral(img1, img2, num_frames)
    else:
        frames = []
        half_frames = num_frames // 2
        frames.extend([img1.astype(np.uint8)] * half_frames)
        frames.extend([img2.astype(np.uint8)] * (num_frames - half_frames))
        return frames

def add_text_to_frame(frame, text, font_path="arial.ttf", font_size=40, color=(255, 255, 255), position=(50, 50)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return np.array(img)

def create_video(frames, fps, frame_size, add_text=False, text_content=""):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = "temp_output.mp4"
    video_writer = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    progress_bar = st.progress(0)
    for i, frame in enumerate(frames):
        if add_text and text_content:
            frame = add_text_to_frame(frame, text_content, position=(50, 50))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        progress_bar.progress((i + 1) / len(frames))
    video_writer.release()
    return temp_output

# --- Streamlit UI ---

st.title("🎥 Frame to Frame by Loop507 - Video Transitions")

with st.sidebar:
    st.header("Impostazioni Input / Output")
    img1_file = st.file_uploader("Carica la prima immagine", type=["png","jpg","jpeg"])
    img2_file = st.file_uploader("Carica la seconda immagine", type=["png","jpg","jpeg"])
    img3_file = st.file_uploader("Carica la terza immagine (opzionale)", type=["png","jpg","jpeg"])
    
    st.header("Parametri Video")
    width = st.number_input("Larghezza video (px)", min_value=64, max_value=1920, value=640)
    height = st.number_input("Altezza video (px)", min_value=64, max_value=1080, value=480)
    fps = st.number_input("FPS (fotogrammi per secondo)", min_value=1, max_value=60, value=30)
    duration = st.number_input("Durata per transizione (secondi)", min_value=0.5, max_value=20.0, value=5.0, step=0.1)
    
    st.header("Effetti Transizione (scegli uno)")
    effect_options = {
        "Sequenza semplice": "simple",
        "🎨 Dissolvenza": "fade",
        "🔍 Zoom": "zoom",
        "🎲 Pixel casuali": "pixel_random",
        "🧩 Blocchi": "pixel_blocks",
        "🌊 Onda": "pixel_wave",
        "🌀 Spirale": "pixel_spiral",
    }
    effect_display = st.radio("Seleziona effetto", list(effect_options.keys()))
    effect_type = effect_options[effect_display]

    block_size = 8
    zoom_factor = 1.5
    if effect_type == "pixel_blocks":
        block_size = st.number_input("Dimensione blocco pixel", min_value=2, max_value=64, value=8)
    if effect_type == "zoom":
        zoom_factor = st.slider("Fattore di zoom", 1.0, 3.0, 1.5)

    add_text = st.checkbox("Aggiungi testo al video")
    text_content = ""
    if add_text:
        text_content = st.text_input("Testo da aggiungere")

st.write("---")

if img1_file and img2_file:
    # Carica immagini da file uploader in PIL
    img1_pil = Image.open(img1_file).convert('RGB')
    img2_pil = Image.open(img2_file).convert('RGB')
    img3_pil = None
    if img3_file:
        img3_pil = Image.open(img3_file).convert('RGB')

    # Ridimensiona immagini
    img1 = np.array(img1_pil.resize((width, height))).astype(np.float32)
    img2 = np.array(img2_pil.resize((width, height))).astype(np.float32)
    if img3_pil:
        img3 = np.array(img3_pil.resize((width, height))).astype(np.float32)
    else:
        img3 = None

    num_transitions = 2 if img3 is not None else 1
    num_frames = int(duration * fps)
    total_duration = calculate_video_duration(num_frames, fps, num_transitions)

    st.write(f"📊 Durata video stimata: **{total_duration:.2f} secondi**")
    st.write(f"📊 Numero di fotogrammi per transizione: **{num_frames}**")
    st.write(f"📊 Effetto selezionato: **{effect_display}**")
    
    if st.button("▶️ Genera Video"):
        all_frames = []
        with st.spinner("Generazione frame..."):
            frames_1_to_2 = generate_transition_frames(img1, img2, num_frames, effect_type, block_size, zoom_factor)
            all_frames.extend(frames_1_to_2)
            if img3 is not None:
                frames_2_to_3 = generate_transition_frames(img2, img3, num_frames, effect_type, block_size, zoom_factor)
                all_frames.extend(frames_2_to_3)
        
        output_path = create_video(all_frames, fps, (width, height), add_text, text_content)
        
        st.success("🎉 Video creato con successo!")
        video_file = open(output_path, 'rb').read()
        st.video(video_file)

else:
    st.info("Carica almeno due immagini per iniziare.")

