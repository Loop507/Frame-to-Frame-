import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
import random
import datetime

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Video Generator by Loop507", layout="wide")

# ─────────────────────────────────────────────
# SMOOTH ALPHA
# ─────────────────────────────────────────────
def smooth_alpha(num_frames, mode="smootherstep"):
    t = np.linspace(0, 1, num_frames)
    if mode == "smoothstep":
        return t * t * (3 - 2 * t)
    elif mode == "smootherstep":
        return t * t * t * (t * (t * 6 - 15) + 10)
    elif mode == "sinusoidal":
        return (1 - np.cos(t * np.pi)) / 2
    else:
        return t  # lineare (fallback)

# ─────────────────────────────────────────────
# EFFETTI
# ─────────────────────────────────────────────
def fade_effect(img1, img2, num_frames, intensity, easing):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return [(img1 * (1 - a) + img2 * a).astype(np.uint8)
            for a in smooth_alpha(num_frames, easing)]

def morph_effect(img1, img2, num_frames, intensity, easing):
    return [cv2.addWeighted(img1, 1 - a, img2, a, 0).astype(np.uint8)
            for a in smooth_alpha(num_frames, easing)]

def glitch_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        for _ in range(intensity):
            y = random.randint(0, h - 2)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], random.randint(-intensity, intensity), axis=1)
        frames.append(frame.astype(np.uint8))
    return frames

def pixelate_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        scale = random.randint(4, 10)
        small = cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(frame)
    return frames

def corrupt_lines_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        for _ in range(intensity):
            x = random.randint(0, w - 15)
            y = random.randint(0, h - 1)
            width = random.randint(5, 15)
            frame[y:y+1, x:x+width] = np.random.randint(0, 255, (1, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames

def wave_distort_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        blend = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        distorted = np.zeros_like(blend)
        for y in range(h):
            offset = int(intensity * np.sin(2 * np.pi * y / 50))
            distorted[y] = np.roll(blend[y], offset, axis=0)
        frames.append(distorted)
    return frames

def zoom_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for i, a in enumerate(smooth_alpha(num_frames, easing)):
        zoom = 1 + (intensity / 100.0) * (i / num_frames)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 0, zoom)
        z1 = cv2.warpAffine(img1, M, (w, h))
        z2 = cv2.warpAffine(img2, M, (w, h))
        frames.append(cv2.addWeighted(z1, 1 - a, z2, a, 0).astype(np.uint8))
    return frames

def zoom_random_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        zoom = random.uniform(1, 1 + intensity / 50.0)
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        z1 = cv2.warpAffine(img1, M, (w, h))
        z2 = cv2.warpAffine(img2, M, (w, h))
        frames.append(cv2.addWeighted(z1, 1 - a, z2, a, 0))
    return frames

def color_echo_effect(img1, img2, num_frames, intensity, easing):
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        r = frame.copy(); g = frame.copy(); b = frame.copy()
        r[:, :, 1:] = 0
        g[:, :, [0, 2]] = 0
        b[:, :, :2] = 0
        merged = cv2.addWeighted(r, 0.5, g, 0.5, 0)
        merged = cv2.addWeighted(merged, 0.5, b, 0.5, 0)
        frames.append(merged.astype(np.uint8))
    return frames

def particle_float_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    particles = [(random.randint(0, w), random.randint(0, h)) for _ in range(50)]
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        for x, y in particles:
            nx = min(w - 1, max(0, x + random.randint(-intensity, intensity)))
            ny = min(h - 1, max(0, y + random.randint(-intensity, intensity)))
            cv2.circle(frame, (nx, ny), 2, (255, 255, 255), -1)
        frames.append(frame)
    return frames

def film_look_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        blend = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        noise = np.random.normal(0, intensity, (h, w, 3)).astype(np.uint8)
        flicker = random.uniform(0.95, 1.05)
        frames.append(cv2.addWeighted(blend, flicker, noise, 0.1, 0))
    return frames

def vhs_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        for _ in range(intensity):
            y = random.randint(0, h - 1)
            frame[y:y+1] = np.roll(frame[y:y+1], random.randint(-5, 5), axis=1)
        frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
        frames.append(frame)
    return frames

def motion_blur_effect(img1, img2, num_frames, intensity, easing):
    k = max(3, intensity)
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k) / k
    frames = []
    for a in smooth_alpha(num_frames, easing):
        blended = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        frames.append(cv2.filter2D(blended, -1, kernel))
    return frames

def scanlines_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    spacing = max(1, 10 - intensity // 5)
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        for y in range(0, h, spacing):
            frame[y:y+1, :] = (frame[y:y+1, :] * 0.5).astype(np.uint8)
        frames.append(frame)
    return frames

def burn_in_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    overlay = np.zeros_like(img1, dtype=np.uint8)
    for i in range(0, h, 20):
        cv2.line(overlay, (0, i), (w, i), (intensity * 5, intensity * 3, intensity * 2), 1)
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        frames.append(cv2.addWeighted(frame, 1, overlay, 0.2, 0))
    return frames

def light_leaks_effect(img1, img2, num_frames, intensity, easing):
    h, w, _ = img1.shape
    frames = []
    for a in smooth_alpha(num_frames, easing):
        frame = cv2.addWeighted(img1, 1 - a, img2, a, 0)
        leak = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(leak, (random.randint(0, w), random.randint(0, h)),
                   random.randint(w // 4, w // 2),
                   (intensity * 5, intensity * 3, intensity * 2), -1)
        leak = cv2.GaussianBlur(leak, (101, 101), 0)
        frames.append(cv2.addWeighted(frame, 1, leak, 0.3, 0))
    return frames

def combo_effect_preset1(img1, img2, num_frames, intensity, easing):
    half = num_frames // 2
    return glitch_effect(img1, img2, half, intensity, easing) + pixelate_effect(img1, img2, half, intensity, easing)

def combo_effect_preset2(img1, img2, num_frames, intensity, easing):
    third = num_frames // 3
    return (wave_distort_effect(img1, img2, third, intensity, easing) +
            color_echo_effect(img1, img2, third, intensity, easing) +
            particle_float_effect(img1, img2, third, intensity, easing))

def combo_effect_preset3(img1, img2, num_frames, intensity, easing):
    third = num_frames // 3
    return (motion_blur_effect(img1, img2, third, intensity, easing) +
            film_look_effect(img1, img2, third, intensity, easing) +
            zoom_random_effect(img1, img2, third, intensity, easing))

# ─────────────────────────────────────────────
# DIZIONARIO EFFETTI
# ─────────────────────────────────────────────
effect_funcs = {
    "Fade": fade_effect,
    "Morph": morph_effect,
    "Glitch": glitch_effect,
    "Pixelate": pixelate_effect,
    "Corrupt Lines": corrupt_lines_effect,
    "Wave Distort": wave_distort_effect,
    "Zoom": zoom_effect,
    "Zoom Random": zoom_random_effect,
    "Color Echo": color_echo_effect,
    "Particle Float": particle_float_effect,
    "Film Look": film_look_effect,
    "VHS": vhs_effect,
    "Motion Blur": motion_blur_effect,
    "Scanlines": scanlines_effect,
    "Burn-in": burn_in_effect,
    "Light Leaks": light_leaks_effect,
    "Combo Creative 1": combo_effect_preset1,
    "Combo Creative 2": combo_effect_preset2,
    "Combo Creative 3": combo_effect_preset3,
}

effect_descriptions = {
    "Fade": "Dissolvenza classica tra un frame e l'altro.",
    "Morph": "Fusione fluida con blending pesato.",
    "Glitch": "Artefatti digitali e distorsioni di riga tipici del segnale corrotto.",
    "Pixelate": "Pixelazione progressiva con scala random per un look 8-bit.",
    "Corrupt Lines": "Righe di dati corrotti che appaiono casualmente sul frame.",
    "Wave Distort": "Distorsione ondulata sinusoidale delle righe orizzontali.",
    "Zoom": "Zoom progressivo dal centro durante la transizione.",
    "Zoom Random": "Zoom da punto focale casuale per un effetto caotico.",
    "Color Echo": "Separazione e fusione dei canali RGB per aloni cromatici.",
    "Particle Float": "Particelle bianche fluttuanti sopra il blend.",
    "Film Look": "Grana cinematografica con flickering di esposizione.",
    "VHS": "Artefatti da nastro magnetico con equalizzazione del canale rosso.",
    "Motion Blur": "Sfocatura da movimento orizzontale sul frame blendato.",
    "Scanlines": "Righe di scansione stile monitor CRT.",
    "Burn-in": "Sovraimpressione di righe con effetto burn phosphor.",
    "Light Leaks": "Aloni di luce colorata proiettati sul frame, stile analogico.",
    "Combo Creative 1": "Glitch + Pixelate in sequenza.",
    "Combo Creative 2": "Wave Distort + Color Echo + Particle Float in sequenza.",
    "Combo Creative 3": "Motion Blur + Film Look + Zoom Random in sequenza.",
}

# ─────────────────────────────────────────────
# REPORT TEMPLATE
# ─────────────────────────────────────────────
def generate_report(params: dict) -> str:
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    music_line = ""
    if params.get("music_enabled") and params.get("bpm"):
        mode_label = "Tutto il brano" if params.get("full_song_mode") else "Durata manuale"
        audio_dur = f"{params['audio_duration']:.1f}s" if params.get("audio_duration") else "—"
        music_line = f"""
🎵 AUDIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File audio      : {params['music_name']}
BPM rilevati    : {params['bpm']:.1f} BPM
Beat interval   : {60 / params['bpm']:.3f} sec
Durata brano    : {audio_dur}
Modalità durata : {mode_label}
Transizioni     : sincronizzate ai beat
"""
    else:
        music_line = "\n🔇 Nessun audio aggiunto al video.\n"

    report = f"""
╔══════════════════════════════════════════╗
║  🎞️  FRAME-TO-FRAME FX  —  by Loop507   ║
╚══════════════════════════════════════════╝

Generato il  : {now}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 PARAMETRI VIDEO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Effetto         : {params['effect']}
Easing          : {params['easing']}
Formato         : {params['format']}
Risoluzione     : {params['resolution']}
Durata          : {params['duration']} secondi
FPS             : {params['fps']}
Frame totali    : {params['total_frames']}
Immagini usate  : {params['num_images']}
Transizioni     : {params['num_transitions']}
Intensità       : {params['intensity']} / 50
{music_line}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 DESCRIZIONE EFFETTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{effect_descriptions.get(params['effect'], 'Effetto personalizzato.')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📣 CAPTION SUGGERITA PER SOCIAL / YOUTUBE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎞️ {params['effect']} transition — {params['format']} format | {params['duration']}s | {params['fps']}fps
Creato con Frame-to-Frame FX by Loop507

Effetto: {params['effect']} ({params['easing']} easing)
{f"🎵 Beat-sync @ {params['bpm']:.1f} BPM — {params['music_name']}" if params.get('music_enabled') and params.get('bpm') else ""}

#VideoArt #FrameToFrame #Loop507 #VisualFX #{params['effect'].replace(' ','')} #TransitionFX #Reels #Shorts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛠️  STACK TECNICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Python · OpenCV · NumPy · imageio · Streamlit
{f"librosa (beat detection) · ffmpeg (audio mux)" if params.get('music_enabled') else ""}

╔══════════════════════════════════════════╗
║         Loop507 — All rights reserved    ║
╚══════════════════════════════════════════╝
"""
    return report.strip()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key in ["video_bytes", "audio_bytes", "audio_name", "report_text", "video_ready"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# INTERFACCIA
# ─────────────────────────────────────────────
st.title("🎞️ Frame-to-Frame FX Video Generator by Loop507")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Carica almeno 2 (massimo 10) immagini",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

with col2:
    effect_choice = st.selectbox("Effetto", list(effect_funcs.keys()))
    easing_choice = st.selectbox("Curva easing", ["smootherstep", "smoothstep", "sinusoidal", "linear"])
    format_choice = st.selectbox("Formato video", ["1:1", "16:9", "9:16"])

col3, col4 = st.columns(2)
with col3:
    intensity = st.slider("Intensità effetto", 1, 50, 15)
    use_music = st.toggle("🎵 Sincronizza musica ai beat", value=False)
    audio_file = None
    if use_music:
        audio_file = st.file_uploader("Carica brano (MP3 / WAV)", type=["mp3", "wav"])
with col4:
    duration_mode = st.radio(
        "Durata video",
        ["⏱️ Durata manuale", "🎵 Tutto il brano"],
        disabled=(not use_music or audio_file is None),
        help="'Tutto il brano' è disponibile solo con un brano caricato"
    )
    duration = st.slider("Durata manuale (secondi)", 1, 60, 5,
                         disabled=(duration_mode == "🎵 Tutto il brano"))

# ─────────────────────────────────────────────
# GENERAZIONE
# ─────────────────────────────────────────────
video_btn = st.button("🎬 Genera Video")

if video_btn and uploaded_files and 2 <= len(uploaded_files) <= 10:

    format_dims = {
        "1:1": (720, 720),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
    }
    target_size = format_dims[format_choice]
    FPS = 24

    images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
    images = [cv2.resize(img, target_size) for img in images]

    num_transitions = len(images) - 1
    bpm_value = None
    music_name = None
    tmp_audio_path = None
    audio_duration_sec = None
    full_song_mode = (duration_mode == "🎵 Tutto il brano" and use_music and audio_file is not None)

    # ── Beat detection & audio save ─────────────────
    if use_music and audio_file is not None:
        try:
            import librosa
            audio_bytes_raw = audio_file.read()
            suffix = os.path.splitext(audio_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
                tmp_audio.write(audio_bytes_raw)
                tmp_audio_path = tmp_audio.name

            # Salva audio in session_state per download separato
            st.session_state["audio_bytes"] = audio_bytes_raw
            st.session_state["audio_name"] = audio_file.name

            y_audio, sr = librosa.load(tmp_audio_path, sr=None)
            tempo, beat_frames = librosa.beat.beat_track(y=y_audio, sr=sr)
            bpm_value = float(tempo)
            music_name = audio_file.name
            audio_duration_sec = len(y_audio) / sr

            beat_interval_sec = 60.0 / bpm_value
            frames_per_transition = max(6, int(beat_interval_sec * FPS))
            st.info(f"🎵 BPM: **{bpm_value:.1f}** — beat ogni **{beat_interval_sec:.2f}s** ({frames_per_transition} frame/transizione) — durata brano: **{audio_duration_sec:.1f}s**")
        except ImportError:
            st.warning("⚠️ librosa non installato. pip install librosa. Procedo senza sync.")
            frames_per_transition = int(FPS * duration / num_transitions)
            full_song_mode = False
        except Exception as e:
            st.warning(f"⚠️ Errore analisi audio: {e}. Procedo senza sync.")
            frames_per_transition = int(FPS * duration / num_transitions)
            full_song_mode = False
    else:
        frames_per_transition = int(FPS * duration / num_transitions)

    # ── Calcola quante transizioni servono ──────────
    if full_song_mode and audio_duration_sec:
        total_video_frames_needed = int(audio_duration_sec * FPS)
        total_transitions_needed = max(num_transitions, int(np.ceil(total_video_frames_needed / frames_per_transition)))
    else:
        total_transitions_needed = num_transitions

    # ── Generazione frame (con ciclo immagini se full_song) ──
    all_frames = []
    progress = st.progress(0.0)
    status = st.empty()

    for i in range(total_transitions_needed):
        idx1 = i % len(images)
        idx2 = (i + 1) % len(images)
        status.text(f"Elaborazione transizione {i+1}/{total_transitions_needed}...")
        frames = effect_funcs[effect_choice](
            images[idx1], images[idx2], frames_per_transition, intensity, easing_choice
        )
        all_frames.extend(frames)
        progress.progress((i + 1) / total_transitions_needed)

        # Stop se in modalità full_song e abbiamo abbastanza frame
        if full_song_mode and audio_duration_sec:
            if len(all_frames) >= int(audio_duration_sec * FPS):
                break

    # Tronca esattamente alla durata del brano se full_song
    if full_song_mode and audio_duration_sec:
        all_frames = all_frames[:int(audio_duration_sec * FPS)]

    status.text("Scrittura video...")

    # ── Scrivi video ────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        video_path = tmpfile.name

    writer = imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    # ── Mux audio se presente ───────────────────────
    final_video_path = video_path
    if use_music and tmp_audio_path and os.path.exists(tmp_audio_path):
        try:
            import subprocess
            muxed_path = video_path.replace(".mp4", "_audio.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", tmp_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                muxed_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                final_video_path = muxed_path
                status.text("✅ Audio aggiunto al video!")
            else:
                st.warning("⚠️ ffmpeg: errore nel mux audio. Video senza audio.")
        except Exception as e:
            st.warning(f"⚠️ Impossibile aggiungere audio: {e}")

    # ── Salva in session_state ──────────────────────
    with open(final_video_path, "rb") as f:
        st.session_state["video_bytes"] = f.read()

    total_frames = len(all_frames)
    actual_duration = total_frames / FPS
    w, h = target_size
    resolution_str = f"{w}×{h}px"

    report_params = {
        "effect": effect_choice,
        "easing": easing_choice,
        "format": format_choice,
        "resolution": resolution_str,
        "duration": f"{actual_duration:.1f}",
        "fps": FPS,
        "total_frames": total_frames,
        "num_images": len(images),
        "num_transitions": total_transitions_needed,
        "intensity": intensity,
        "music_enabled": use_music and audio_file is not None,
        "bpm": bpm_value,
        "music_name": music_name,
        "full_song_mode": full_song_mode,
        "audio_duration": audio_duration_sec,
    }
    st.session_state["report_text"] = generate_report(report_params)
    st.session_state["video_ready"] = True

    status.empty()
    progress.empty()
    st.success("✅ Video generato con successo!")

elif video_btn:
    st.warning("⚠️ Carica tra 2 e 10 immagini prima di generare.")

# ─────────────────────────────────────────────
# DOWNLOAD — sempre visibili dopo la generazione
# (FUORI da st.button → non si resettano al rerun)
# ─────────────────────────────────────────────
if st.session_state.get("video_ready"):
    st.divider()
    has_audio = st.session_state.get("audio_bytes") is not None
    dl_cols = st.columns(3 if has_audio else 2)

    with dl_cols[0]:
        st.download_button(
            label="📥 Scarica Video (.mp4)",
            data=st.session_state["video_bytes"],
            file_name="loop507_output.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    with dl_cols[1]:
        st.download_button(
            label="📄 Scarica Report (.txt)",
            data=st.session_state["report_text"],
            file_name="loop507_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if has_audio:
        audio_ext = os.path.splitext(st.session_state["audio_name"])[1]
        with dl_cols[2]:
            st.download_button(
                label=f"🎵 Scarica Audio ({audio_ext})",
                data=st.session_state["audio_bytes"],
                file_name=st.session_state["audio_name"],
                mime="audio/mpeg" if audio_ext == ".mp3" else "audio/wav",
                use_container_width=True,
            )

    with st.expander("👁️ Anteprima report"):
        st.code(st.session_state["report_text"], language=None)
else:
    st.info("Carica almeno 2 immagini e premi 'Genera Video'.")
