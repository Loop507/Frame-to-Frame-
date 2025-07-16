import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import argparse
import os
from tqdm import tqdm
import random

# --- Funzioni di effetto ---

def load_image(path, size):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(size)
        return np.array(img).astype(np.float32)
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {path}")
    except Exception as e:
        raise Exception(f"Errore nel caricamento dell'immagine: {e}")

def calculate_video_duration(frames, fps, num_transitions=1):
    """Calcola la durata del video in secondi"""
    total_frames = frames * num_transitions
    return total_frames / fps

def fade_effect(img1, img2, num_frames):
    frames = []
    for i in tqdm(range(num_frames), desc="Dissolvenza"):
        alpha = i / float(num_frames - 1)
        blended = (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
        frames.append(blended)
    return frames

def zoom_effect(img1, img2, num_frames, zoom_factor=1.2):
    frames = []
    h, w = img1.shape[:2]
    
    for i in tqdm(range(num_frames), desc="Zoom"):
        alpha = i / float(num_frames - 1)
        scale = 1 + (zoom_factor - 1) * alpha
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Evita che l'immagine diventi troppo piccola
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
    """Scambia pixel casuali tra le due immagini"""
    frames = []
    h, w, c = img1.shape
    total_pixels = h * w
    
    for i in tqdm(range(num_frames), desc="Pixel Swap Random"):
        alpha = i / float(num_frames - 1)
        num_swapped = int(total_pixels * alpha)
        
        # Crea una copia dell'immagine 1
        frame = img1.copy()
        
        # Genera coordinate casuali per lo scambio
        swap_coords = []
        for _ in range(num_swapped):
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)
            swap_coords.append((y, x))
        
        # Scambia i pixel
        for y, x in swap_coords:
            frame[y, x] = img2[y, x]
        
        frames.append(frame.astype(np.uint8))
    
    return frames

def pixel_swap_blocks(img1, img2, num_frames, block_size=8):
    """Scambia blocchi di pixel tra le due immagini"""
    frames = []
    h, w, c = img1.shape
    
    # Calcola il numero di blocchi
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w
    
    for i in tqdm(range(num_frames), desc="Pixel Swap Blocks"):
        alpha = i / float(num_frames - 1)
        num_swapped_blocks = int(total_blocks * alpha)
        
        # Crea una copia dell'immagine 1
        frame = img1.copy()
        
        # Genera coordinate casuali per i blocchi
        block_coords = []
        for _ in range(num_swapped_blocks):
            by = random.randint(0, blocks_h-1)
            bx = random.randint(0, blocks_w-1)
            block_coords.append((by, bx))
        
        # Scambia i blocchi
        for by, bx in block_coords:
            y_start = by * block_size
            y_end = min(y_start + block_size, h)
            x_start = bx * block_size
            x_end = min(x_start + block_size, w)
            
            frame[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
        
        frames.append(frame.astype(np.uint8))
    
    return frames

def pixel_swap_wave(img1, img2, num_frames):
    """Scambia pixel con effetto onda da sinistra a destra"""
    frames = []
    h, w, c = img1.shape
    
    for i in tqdm(range(num_frames), desc="Pixel Swap Wave"):
        alpha = i / float(num_frames - 1)
        
        # Crea una copia dell'immagine 1
        frame = img1.copy()
        
        # Calcola la posizione dell'onda
        wave_pos = int(w * alpha)
        
        for y in range(h):
            # Aggiunge un leggero effetto ondulatorio
            wave_offset = int(10 * np.sin(y * 0.1 + i * 0.3))
            x_threshold = wave_pos + wave_offset
            
            # Scambia i pixel a sinistra della soglia
            if x_threshold > 0:
                end_x = min(x_threshold, w)
                frame[y, :end_x] = img2[y, :end_x]
        
        frames.append(frame.astype(np.uint8))
    
    return frames

def pixel_swap_spiral(img1, img2, num_frames):
    """Scambia pixel con effetto spirale dal centro"""
    frames = []
    h, w, c = img1.shape
    center_y, center_x = h // 2, w // 2
    
    # Genera le coordinate in ordine spirale
    spiral_coords = []
    max_radius = int(np.sqrt(center_x**2 + center_y**2))
    
    for radius in range(max_radius + 1):
        for angle in np.linspace(0, 2*np.pi, max(8, radius*2), endpoint=False):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                spiral_coords.append((y, x))
    
    for i in tqdm(range(num_frames), desc="Pixel Swap Spiral"):
        alpha = i / float(num_frames - 1)
        num_swapped = int(len(spiral_coords) * alpha)
        
        # Crea una copia dell'immagine 1
        frame = img1.copy()
        
        # Scambia i pixel seguendo la spirale
        for j in range(num_swapped):
            if j < len(spiral_coords):
                y, x = spiral_coords[j]
                frame[y, x] = img2[y, x]
        
        frames.append(frame.astype(np.uint8))
    
    return frames

def generate_transition_frames(img1, img2, num_frames, effect_type, block_size=8, zoom_factor=1.2):
    """Genera frame di transizione tra due immagini"""
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
        # Effetto semplice: mostra img1 per metà frame, img2 per l'altra metà
        frames = []
        half_frames = num_frames // 2
        for i in tqdm(range(half_frames), desc="Transizione semplice"):
            frames.append(img1.astype(np.uint8))
        for i in tqdm(range(num_frames - half_frames), desc="Transizione semplice"):
            frames.append(img2.astype(np.uint8))
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

# --- Genera il video ---

def create_video(frames, output_path, fps, frame_size, add_text=False, text_content=""):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    print("💾 Salvando video...")
    for i, frame in enumerate(tqdm(frames, desc="Scrittura video")):
        if add_text and text_content:
            frame = add_text_to_frame(frame, text_content, position=(50, 50))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"✅ Video salvato: {output_path}")

# --- Validazione input ---

def validate_args(args):
    if not os.path.exists(args.img1):
        raise FileNotFoundError(f"Prima immagine non trovata: {args.img1}")
    if not os.path.exists(args.img2):
        raise FileNotFoundError(f"Seconda immagine non trovata: {args.img2}")
    if args.img3 and not os.path.exists(args.img3):
        raise FileNotFoundError(f"Terza immagine non trovata: {args.img3}")
    if args.frames <= 0 or args.fps <= 0:
        raise ValueError("Frames e FPS devono essere positivi")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Larghezza e altezza devono essere positive")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Frame to Frame by Loop507 - Effetti avanzati con scambio pixel e supporto 3 foto")
    parser.add_argument("--img1", type=str, required=True, help="Percorso della prima immagine")
    parser.add_argument("--img2", type=str, required=True, help="Percorso della seconda immagine")
    parser.add_argument("--img3", type=str, help="Percorso della terza immagine (opzionale)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Nome del video di output")
    parser.add_argument("--frames", type=int, default=150, help="Numero di fotogrammi per transizione (default: 150 = 5 sec a 30fps)")
    parser.add_argument("--fps", type=int, default=30, help="Fotogrammi al secondo")
    parser.add_argument("--duration", type=float, help="Durata in secondi per transizione (sovrascrive --frames)")
    parser.add_argument("--width", type=int, default=640, help="Larghezza del video")
    parser.add_argument("--height", type=int, default=480, help="Altezza del video")
    parser.add_argument("--fade", action="store_true", help="Attiva la dissolvenza")
    parser.add_argument("--zoom", action="store_true", help="Attiva lo zoom progressivo")
    parser.add_argument("--pixel-random", action="store_true", help="Scambio pixel casuali")
    parser.add_argument("--pixel-blocks", action="store_true", help="Scambio a blocchi")
    parser.add_argument("--pixel-wave", action="store_true", help="Scambio con effetto onda")
    parser.add_argument("--pixel-spiral", action="store_true", help="Scambio con effetto spirale")
    parser.add_argument("--block-size", type=int, default=8, help="Dimensione blocchi per pixel-blocks")
    parser.add_argument("--zoom-factor", type=float, default=1.5, help="Fattore di zoom (default: 1.5)")
    parser.add_argument("--text", type=str, default="", help="Aggiungi un testo al video")
    args = parser.parse_args()

    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Errore: {e}")
        return

    # Se è specificata la durata, calcola i frame
    if args.duration:
        args.frames = int(args.duration * args.fps)
        print(f"🎬 Durata impostata: {args.duration} secondi = {args.frames} frame")

    # Determina il numero di transizioni
    num_transitions = 2 if args.img3 else 1
    total_duration = calculate_video_duration(args.frames, args.fps, num_transitions)
    
    print(f"📊 Informazioni video:")
    print(f"   • Frame per transizione: {args.frames}")
    print(f"   • FPS: {args.fps}")
    print(f"   • Transizioni: {num_transitions}")
    print(f"   • Durata totale: {total_duration:.2f} secondi")
    print(f"   • Dimensioni: {args.width}x{args.height}")
    print()

    # Determina il tipo di effetto
    effect_type = "simple"
    effect_name = "Sequenza semplice"
    if args.pixel_spiral:
        effect_type = "pixel_spiral"
        effect_name = "🌀 Spirale"
    elif args.pixel_wave:
        effect_type = "pixel_wave"
        effect_name = "🌊 Onda"
    elif args.pixel_blocks:
        effect_type = "pixel_blocks"
        effect_name = f"🧩 Blocchi ({args.block_size}x{args.block_size})"
    elif args.pixel_random:
        effect_type = "pixel_random"
        effect_name = "🎲 Pixel casuali"
    elif args.fade and args.zoom:
        effect_type = "zoom"
        effect_name = "🎨🔍 Zoom con dissolvenza"
    elif args.fade:
        effect_type = "fade"
        effect_name = "🎨 Dissolvenza"
    elif args.zoom:
        effect_type = "zoom"
        effect_name = "🔍 Zoom"
    
    print(f"🎞️ Effetto selezionato: {effect_name}")

    print("🖼️ Caricamento immagini...")
    try:
        img1 = load_image(args.img1, (args.width, args.height))
        img2 = load_image(args.img2, (args.width, args.height))
        img3 = None
        if args.img3:
            img3 = load_image(args.img3, (args.width, args.height))
    except Exception as e:
        print(f"❌ Errore nel caricamento: {e}")
        return

    print("🎬 Generazione video...")
    all_frames = []
    
    # Prima transizione: img1 -> img2
    print("\n📽️ Transizione 1/1:" if not args.img3 else "\n📽️ Transizione 1/2:")
    frames_1_to_2 = generate_transition_frames(img1, img2, args.frames, effect_type, args.block_size, args.zoom_factor)
    all_frames.extend(frames_1_to_2)
    
    # Seconda transizione: img2 -> img3 (se presente)
    if args.img3:
        print("\n📽️ Transizione 2/2:")
        frames_2_to_3 = generate_transition_frames(img2, img3, args.frames, effect_type, args.block_size, args.zoom_factor)
        all_frames.extend(frames_2_to_3)

    create_video(all_frames, args.output, args.fps, (args.width, args.height),
                 add_text=bool(args.text), text_content=args.text)
    
    print(f"\n🎉 Video completato!")
    print(f"   • File: {args.output}")
    print(f"   • Durata: {len(all_frames) / args.fps:.2f} secondi")
    print(f"   • Frame totali: {len(all_frames)}")
    print(f"   • Effetto: {effect_name}")
