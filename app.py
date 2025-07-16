import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import os

# --- Effetti ---

def fade_transition(img1, img2, num_frames):
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        blended = (arr1 * (1 - alpha) + arr2 * alpha).astype(np.uint8)
        frames.append(blended)
    return frames

def zoom_transition(img1, img2, num_frames, zoom_factor=1.5):
    w, h = img1.size
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        scale = 1 + (zoom_factor - 1) * alpha
        
        new_w, new_h = int(w * scale), int(h * scale)
        img1_zoomed = img1.resize((new_w, new_h), Image.LANCZOS)
        img2_zoomed = img2.resize((new_w, new_h), Image.LANCZOS)
        
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img1_crop = img1_zoomed.crop((left, top, left + w, top + h))
        img2_crop = img2_zoomed.crop((left, top, left + w, top + h))
        
        arr1_crop = np.array(img1_crop).astype(np.float32)
        arr2_crop = np.array(img2_crop).astype(np.float32)
        
        blended = (arr1_crop * (1 - alpha) + arr2_crop * alpha).astype(np.uint8)
        frames.append(blended)
    return frames

def pixel_random_transition(img1, img2, num_frames):
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)
    h, w, c = arr1.shape
    total_pixels = h * w
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swap = int(total_pixels * alpha)
        
        frame = arr1.copy()
        ys = np.random.randint(0, h, num_swap)
        xs = np.random.randint(0, w, num_swap)
        frame[ys, xs] = arr2[ys, xs]
        frames.append(frame)
    return frames

def pixel_block_transition(img1, img2, num_frames, block_size=8):
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)
    h, w, c = arr1.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w
    frames = []

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swap = int(total_blocks * alpha)

        frame = arr1.copy()
        swapped_blocks = set()
        while len(swapped_blocks) < num_swap:
            by = np.random.randint(0, blocks_h)
            bx = np.random.randint(0, blocks_w)
            if (by, bx) not in swapped_blocks:
                swapped_blocks.add((by, bx))
                y_start = by * block_size
                x_start = bx * block_size
                y_end = min(y_start + block_size, h)
                x_end = min(x_start + block_size, w)
                frame[y_start:y_end, x_start:x_end] = arr2[y_start:y_end, x_start:x_end]
        frames.append(frame)
    return frames

def wave_transition(img1, img2, num_frames):
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)
    h, w, c = arr1.shape
    frames = []

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = arr1.copy()
        wave_pos = int(w * alpha)
        for y in range(h):
            wave_offset = int(10 * np.sin(y * 0.1 + i * 0.3))
            threshold = wave_pos + wave_offset
            if threshold > 0:
                end_x = min(threshold, w)
                frame[y, :end_x] = arr2[y, :end_x]
        frames.append(frame)
    return frames

def spiral_transition(img1, img2, num_frames):
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)
    h, w, c = arr1.shape
    center_y, center_x = h // 2, w // 2
    max_radius = int(np.sqrt(center_x**2 + center_y**2))
    coords = []

    for radius in range(max_radius + 1):
        points = max(8, radius*2)
        for angle in np.linspace(0, 2*np.pi, points, endpoint=False):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                coords.append((y, x))

    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        num_swap = int(len(coords) * alpha)
        frame = arr1.copy()
        for j in range(num_swap):
            y, x = coords[j]
            frame[y, x] = arr2[y, x]
        frames.append(frame)
    return frames

def slide_transition(img1, img2, num_frames, direction='left'):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    h, w, c = arr1.shape
    frames = []

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = np.zeros_like(arr1)
        if direction == 'left':
            offset = int(w * alpha)
            frame[:, :w-offset] = arr1[:, offset:]
            frame[:, w-offset:] = arr2[:, :offset]
        elif direction == 'right':
            offset = int(w * alpha)
            frame[:, offset:] = arr1[:, :w-offset]
            frame[:, :offset] = arr2[:, w-offset:]
        elif direction == 'up':
            offset = int(h * alpha)
            frame[:h-offset, :] = arr1[offset:, :]
            frame[h-offset:, :] = arr2[:offset, :]
        elif direction == 'down':
            offset = int(h * alpha)
            frame[offset:, :] = arr1[:h-offset, :]
            frame[:offset, :] = arr2[h-offset:, :]
        frames.append(frame)
    return frames

# --- Aggiungi testo ai frame ---

def add_text_to_frame(frame_np, text, font_size=40, color=(255,255,255), position=(10,10)):
    img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return np.array(img)

# --- Streamlit App ---

st.title("🎥 Video MP4 con molti effetti di transizione")

img1_file = st.file_uploader("Carica prima immagine", type=["png","jpg","jpeg"])
img2_file = st.file_uploader("Carica seconda immagine", type=["png","jpg","jpeg"])

width = st.number_input("Larghezza video (px)", min_value=64, max_value=1920, value=640)
height = st.number_input("Altezza video (px)", min_value=64, max_value=1080, value=480)
fps = st.slider("FPS", min_value=1, max_value=60, value=30)
duration = st.slider("Durata transizione (sec)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)

effects = {
    "Dissolvenza (Fade)": fade_transition,
    "Zoom": zoom_transition,
    "Pixel Random": pixel_random_transition,
    "Pixel a Blocchi": pixel_block_transition,
    "Onda": wave_transition,
    "Spirale": spiral_transition,
    "Scorrimento Sinistra": lambda i1,i2,n: slide_transition(i1,i2,n,'left'),
    "Scorrimento Destra": lambda i1,i2,n: slide_transition(i1,i2,n,'right'),
    "Scorrimento Su": lambda i1,i2,n: slide_transition(i1,i2,n,'up'),
    "Scorrimento Giù": lambda i1,i2,n: slide_transition(i1,i2,n,'down'),
}

effect_name = st.selectbox("Scegli effetto di transizione", list(effects.keys()))
effect_func = effects[effect_name]

zoom_factor = 1.5
if effect_name == "Zoom":
    zoom_factor = st.slider("Fattore zoom", 1.0, 3.0, 1.5)
    
block_size = 8
if effect_name == "Pixel a Blocchi":
    block_size = st.slider("Dimensione blocco pixel", 2, 64, 8)

add_text = st.checkbox("Aggiungi testo sul video")
text_content = ""
if add_text:
    text_content = st.text_input("Testo da aggiungere")

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB").resize((width, height))
    img2 = Image.open(img2_file).convert("RGB").resize((width, height))

    if st.button("Genera video MP4"):
        st.info("Generazione video in corso...")
        num_frames = int(duration * fps)

        if effect_name == "Zoom":
            frames = effect_func(img1, img2, num_frames, zoom_factor=zoom_factor)
        elif effect_name == "Pixel a Blocchi":
            frames = effect_func(img1, img2, num_frames, block_size=block_size)
        else:
            frames = effect_func(img1, img2, num_frames)

        if add_text and text_content.strip():
            frames = [add_text_to_frame(f, text_content) for f in frames]

        clip = ImageSequenceClip([Image.fromarray(f) for f in frames], fps=fps)
        output_path = "output.mp4"
        clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)
        st.success("Video generato con successo!")

        if os.path.exists(output_path):
            os.remove(output_path)
