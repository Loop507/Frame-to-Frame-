import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio.v2 as imageio  # API legacy get_writer: la v3 la dichiara obsoleta
import os
import tempfile
import random
import datetime
import subprocess

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Video Generator 3.1 by Loop507", layout="wide")

# ─────────────────────────────────────────────
# ALPHA / EASING
# Gli alpha vanno da 0 (incluso) a 1 (ESCLUSO): l'ultimo frame di una transizione
# non ripete img2, perché il primo frame della successiva parte già da img2.
# → niente frame duplicato alla giunzione, e il loop ciclico è senza stacco.
# ─────────────────────────────────────────────
def smooth_alpha(num_frames, mode="smootherstep"):
    t = np.linspace(0, 1, max(1, int(num_frames)), endpoint=False)
    if mode == "smoothstep":
        return t * t * (3 - 2 * t)
    elif mode == "smootherstep":
        return t * t * t * (t * (t * 6 - 15) + 10)
    elif mode == "sinusoidal":
        return (1 - np.cos(t * np.pi)) / 2
    return t  # lineare (fallback)

def _alphas(num_frames, easing, alphas=None):
    """Lista di float Python (NumPy 2: uno scalare np.float64 promuoverebbe a float64
    gli array float32). Se 'alphas' è dato (Combo) la lunghezza è quella della fetta."""
    arr = smooth_alpha(num_frames, easing) if alphas is None else np.asarray(alphas, dtype=np.float64)
    return [float(x) for x in arr]

def _blend(img1, img2, a):
    return cv2.addWeighted(img1, 1.0 - a, img2, a, 0)

def _env(a):
    """Inviluppo 0→1→0 sulla transizione: gli effetti 'a impulso' partono e finiscono a riposo."""
    return float(np.sin(np.pi * a))

# ─────────────────────────────────────────────
# EFFETTI — tutti generatori (yield): un frame alla volta, niente liste in RAM.
# Firma comune: (img1, img2, num_frames, intensity, easing, alphas=None)
# ─────────────────────────────────────────────
def fade_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    for a in _alphas(num_frames, easing, alphas):
        yield (f1 * (1 - a) + f2 * a).astype(np.uint8)

def morph_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    """Morph vero con optical flow Farneback (DSP puro): i pixel di img1 scivolano verso
    img2 e viceversa, poi crossfade. Il flusso si calcola UNA volta a metà risoluzione."""
    h, w = img1.shape[:2]
    scale = 0.5 if max(h, w) > 480 else 1.0
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    g1 = cv2.cvtColor(cv2.resize(img1, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    params = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    flow12 = cv2.calcOpticalFlowFarneback(g1, g2, None, **params)
    flow21 = cv2.calcOpticalFlowFarneback(g2, g1, None, **params)

    def _up(flow):
        flow = cv2.GaussianBlur(flow, (0, 0), 3)
        return cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale

    f12, f21 = _up(flow12), _up(flow21)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for a in _alphas(num_frames, easing, alphas):
        w1 = cv2.remap(img1, xs - a * f12[..., 0], ys - a * f12[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w2 = cv2.remap(img2, xs - (1 - a) * f21[..., 0], ys - (1 - a) * f21[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        yield _blend(w1, w2, a)

def glitch_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        for _ in range(intensity):
            y = random.randint(0, h - 2)
            frame[y:y+2, :, :] = np.roll(frame[y:y+2, :, :], random.randint(-intensity, intensity), axis=1)
        yield frame

def pixelate_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        scale = random.randint(4, 10)
        small = cv2.resize(frame, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_LINEAR)
        yield cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def corrupt_lines_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        for _ in range(intensity):
            x = random.randint(0, w - 15)
            y = random.randint(0, h - 1)
            width = random.randint(5, 15)
            frame[y:y+1, x:x+width] = np.random.randint(0, 255, (1, width, 3), dtype=np.uint8)
        yield frame

def wave_distort_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    # offset funzione della sola riga → righe con lo stesso offset si ruotano insieme in un colpo
    y_idx = np.arange(h)
    offsets = (intensity * np.sin(2 * np.pi * y_idx / 50)).astype(np.int64)
    groups = [(int(v), np.where(offsets == v)[0]) for v in np.unique(offsets)]
    for a in _alphas(num_frames, easing, alphas):
        blend = _blend(img1, img2, a)
        distorted = np.empty_like(blend)
        for v, rows in groups:
            distorted[rows] = np.roll(blend[rows], v, axis=1)
        yield distorted

def zoom_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    """Punch-in: lo zoom sale e RIENTRA a 1× a fine transizione (nessun 'pop' alla giunzione)."""
    h, w = img1.shape[:2]
    center = (w / 2.0, h / 2.0)
    for a in _alphas(num_frames, easing, alphas):
        zoom = 1.0 + (intensity / 100.0) * _env(a)
        M = cv2.getRotationMatrix2D(center, 0, zoom)
        z1 = cv2.warpAffine(img1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        z2 = cv2.warpAffine(img2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        yield _blend(z1, z2, a)

def zoom_random_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        zoom = 1.0 + random.uniform(0, intensity / 50.0) * _env(a)
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        z1 = cv2.warpAffine(img1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        z2 = cv2.warpAffine(img2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        yield _blend(z1, z2, a)

def _mask_blend(img1, img2, alpha_map):
    return (img1.astype(np.float32) * (1 - alpha_map) + img2.astype(np.float32) * alpha_map).astype(np.uint8)

def wipe_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    """Wipe direzionale orizzontale (sinistra -> destra) con bordo sfumato."""
    h, w = img1.shape[:2]
    feather = max(4, int(intensity))
    xs = np.arange(w)
    for a in _alphas(num_frames, easing, alphas):
        edge = -feather + a * (w + feather)
        row = np.clip((edge - xs) / feather + 1, 0, 1).astype(np.float32)
        yield _mask_blend(img1, img2, row[np.newaxis, :, np.newaxis])

def diagonal_wipe_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    """Wipe diagonale (alto-sx -> basso-dx) con bordo sfumato."""
    h, w = img1.shape[:2]
    feather = max(8, int(intensity) * 2)
    yy, xx = np.mgrid[0:h, 0:w]
    diag = (xx + yy).astype(np.float32)
    max_diag = w + h
    for a in _alphas(num_frames, easing, alphas):
        edge = -feather + a * (max_diag + feather)
        yield _mask_blend(img1, img2, np.clip((edge - diag) / feather + 1, 0, 1)[:, :, np.newaxis])

def iris_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    """Iris circolare che si espande dal centro con bordo sfumato."""
    h, w = img1.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    feather = max(4, int(intensity))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    max_r = float(np.sqrt(cx ** 2 + cy ** 2))
    for a in _alphas(num_frames, easing, alphas):
        radius = -feather + a * (max_r + feather)
        yield _mask_blend(img1, img2, np.clip((radius - dist) / feather + 1, 0, 1)[:, :, np.newaxis])

def color_echo_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    """Aberrazione cromatica: R e B scivolano in direzioni opposte (G fermo) e si ricompongono.
    Lo scarto è un impulso (sin πa): a inizio/fine transizione i canali sono allineati."""
    h, w = img1.shape[:2]
    max_shift = max(1.0, intensity * 0.6)
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        s = max_shift * _env(a)
        r, g, b = cv2.split(frame)
        r = cv2.warpAffine(r, np.float32([[1, 0, s], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
        b = cv2.warpAffine(b, np.float32([[1, 0, -s], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
        yield cv2.merge([r, g, b])

def particle_float_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    particles = [(random.randint(0, w), random.randint(0, h)) for _ in range(50)]
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        for x, y in particles:
            nx = min(w - 1, max(0, x + random.randint(-intensity, intensity)))
            ny = min(h - 1, max(0, y + random.randint(-intensity, intensity)))
            cv2.circle(frame, (nx, ny), 2, (255, 255, 255), -1)
        yield frame

def film_look_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        blend = _blend(img1, img2, a)
        noise = np.clip(np.random.normal(128, intensity, (h, w, 3)), 0, 255).astype(np.uint8)
        flicker = random.uniform(0.95, 1.05)
        # noise centrato su 128: sottraiamo l'offset per ottenere grana simmetrica
        yield cv2.addWeighted(blend, flicker, noise, 0.12, -128 * 0.12)

def vhs_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        for _ in range(intensity):
            y = random.randint(0, h - 1)
            frame[y:y+1] = np.roll(frame[y:y+1], random.randint(-5, 5), axis=1)
        frame[:, :, 0] = cv2.equalizeHist(np.ascontiguousarray(frame[:, :, 0]))
        yield frame

def motion_blur_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    k = max(3, intensity) | 1  # kernel dispari → centrato
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    for a in _alphas(num_frames, easing, alphas):
        yield cv2.filter2D(_blend(img1, img2, a), -1, kernel)

def scanlines_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    spacing = max(2, 10 - intensity // 5)  # min 2: a intensità alta non scurisce TUTTE le righe
    mask = np.ones((h, 1, 1), dtype=np.float32)
    mask[::spacing] = 0.5
    for a in _alphas(num_frames, easing, alphas):
        yield (_blend(img1, img2, a).astype(np.float32) * mask).astype(np.uint8)

def burn_in_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    overlay = np.zeros_like(img1, dtype=np.uint8)
    for i in range(0, h, 20):
        cv2.line(overlay, (0, i), (w, i), (intensity * 5, intensity * 3, intensity * 2), 1)
    for a in _alphas(num_frames, easing, alphas):
        yield cv2.addWeighted(_blend(img1, img2, a), 1, overlay, 0.2, 0)

def light_leaks_effect(img1, img2, num_frames, intensity, easing, alphas=None):
    h, w = img1.shape[:2]
    # glow disegnato e sfocato a 1/4 di risoluzione, poi ingrandito (molto più economico)
    small_w, small_h = max(1, w // 4), max(1, h // 4)
    for a in _alphas(num_frames, easing, alphas):
        frame = _blend(img1, img2, a)
        leak_small = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        cv2.circle(leak_small,
                   (random.randint(0, small_w), random.randint(0, small_h)),
                   random.randint(small_w // 4, small_w // 2) or 1,
                   (intensity * 5, intensity * 3, intensity * 2), -1)
        leak_small = cv2.GaussianBlur(leak_small, (25, 25), 0)
        leak = cv2.resize(leak_small, (w, h), interpolation=cv2.INTER_LINEAR)
        yield cv2.addWeighted(frame, 1, leak, 0.3, 0)

# ── Combo: UNA curva alpha globale tagliata in fette → continuità + frame esatti ──
def _combo(funcs, img1, img2, num_frames, intensity, easing, alphas):
    full = _alphas(num_frames, easing, alphas)
    for fn, part in zip(funcs, np.array_split(np.asarray(full), len(funcs))):
        if len(part) == 0:
            continue
        yield from fn(img1, img2, len(part), intensity, easing, alphas=part)

def combo_effect_preset1(img1, img2, num_frames, intensity, easing, alphas=None):
    yield from _combo((glitch_effect, pixelate_effect), img1, img2, num_frames, intensity, easing, alphas)

def combo_effect_preset2(img1, img2, num_frames, intensity, easing, alphas=None):
    yield from _combo((wave_distort_effect, color_echo_effect, particle_float_effect),
                      img1, img2, num_frames, intensity, easing, alphas)

def combo_effect_preset3(img1, img2, num_frames, intensity, easing, alphas=None):
    yield from _combo((motion_blur_effect, film_look_effect, zoom_random_effect),
                      img1, img2, num_frames, intensity, easing, alphas)

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
    "Wipe Direzionale": wipe_effect,
    "Wipe Diagonale": diagonal_wipe_effect,
    "Iris Circolare": iris_effect,
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
    "Morph": "Morph a optical flow (Farneback): i pixel scivolano da un'immagine all'altra, poi crossfade. DSP puro.",
    "Glitch": "Artefatti digitali e distorsioni di riga tipici del segnale corrotto.",
    "Pixelate": "Pixelazione progressiva con scala random per un look 8-bit.",
    "Corrupt Lines": "Righe di dati corrotti che appaiono casualmente sul frame.",
    "Wave Distort": "Distorsione ondulata sinusoidale delle righe orizzontali.",
    "Zoom": "Punch-in dal centro che rientra a 1x a fine transizione.",
    "Zoom Random": "Zoom da punto focale casuale per un effetto caotico, rientra a fine transizione.",
    "Wipe Direzionale": "Spazzata orizzontale sinistra→destra con bordo sfumato.",
    "Wipe Diagonale": "Spazzata diagonale angolo alto-sx → basso-dx con bordo sfumato.",
    "Iris Circolare": "Apertura circolare dal centro verso i bordi, stile iris cinema.",
    "Color Echo": "Aberrazione cromatica: canali R e B separati in direzioni opposte.",
    "Particle Float": "Particelle bianche fluttuanti sopra il blend.",
    "Film Look": "Grana cinematografica con flickering di esposizione.",
    "VHS": "Artefatti da nastro magnetico con equalizzazione del canale rosso.",
    "Motion Blur": "Sfocatura da movimento orizzontale sul frame blendato.",
    "Scanlines": "Righe di scansione stile monitor CRT.",
    "Burn-in": "Sovraimpressione di righe con effetto burn phosphor.",
    "Light Leaks": "Aloni di luce colorata proiettati sul frame, stile analogico.",
    "Combo Creative 1": "Glitch + Pixelate in sequenza (curva alpha continua).",
    "Combo Creative 2": "Wave Distort + Color Echo + Particle Float in sequenza.",
    "Combo Creative 3": "Motion Blur + Film Look + Zoom Random in sequenza.",
}

# ─────────────────────────────────────────────
# UTILITY: CROP-TO-FILL
# ─────────────────────────────────────────────
def resize_crop_fill(img, target_w, target_h):
    """Ritaglia al centro per rispettare l'aspect ratio del formato scelto, poi ridimensiona."""
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = max(1, int(round(h * target_ratio)))
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        new_h = max(1, int(round(w / target_ratio)))
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

# ─────────────────────────────────────────────
# UTILITY: SEQUENZA IMMAGINI (ciclica o ping-pong)
# ─────────────────────────────────────────────
def build_image_sequence(n_images, loop_mode):
    """'ciclico': 0,1,..,n-1,0,1,..   'ping_pong': 0,1,..,n-1,n-2,..,1,0,1,.."""
    if loop_mode == "ping_pong" and n_images > 2:
        return list(range(n_images)) + list(range(n_images - 2, 0, -1))
    return list(range(n_images))

def get_pair_indices(i, sequence):
    seq_len = len(sequence)
    return sequence[i % seq_len], sequence[(i + 1) % seq_len]

# ─────────────────────────────────────────────
# MODALITA' MISTA
# ─────────────────────────────────────────────
MIXED_LABEL = "🎲 Sequenza mista (random per transizione)"
MIXED_POOL = [k for k in effect_funcs.keys() if not k.startswith("Combo")]

def pick_effect_for_transition(effect_choice, rnd):
    if effect_choice == MIXED_LABEL:
        return rnd.choice(MIXED_POOL)
    return effect_choice

# ─────────────────────────────────────────────
# BEAT GRID → NUMERO DI FRAME PER TRANSIZIONE (senza drift)
# ─────────────────────────────────────────────
GRID_LABELS = {
    "×1 (beat rilevati)": "normal",
    "×½ (una transizione ogni 2 beat)": "half",
    "×2 (suddividi ogni beat)": "double",
}

def adjust_beat_grid(beat_times, mode):
    """Corregge gli errori d'ottava del beat tracker (tempo dimezzato/raddoppiato)."""
    bt = np.asarray(beat_times, dtype=np.float64)
    if len(bt) < 3 or mode == "normal":
        return bt
    if mode == "half":
        out = bt[::2]
        return out if out[-1] == bt[-1] else np.append(out, bt[-1])
    if mode == "double":
        return np.sort(np.concatenate([bt, (bt[:-1] + bt[1:]) / 2.0]))
    return bt

def beat_frame_counts(beat_times, fps, min_frames=4):
    """Frame per transizione dai tempi ASSOLUTI dei beat (round sui confini cumulativi):
    l'errore non si accumula, a differenza di round() su ogni intervallo."""
    edges = np.round(np.asarray(beat_times, dtype=np.float64) * fps).astype(int)
    kept = [int(edges[0])]
    for e in edges[1:]:
        if e - kept[-1] >= min_frames:
            kept.append(int(e))
    if kept[-1] != int(edges[-1]):
        kept[-1] = int(edges[-1])  # l'ultimo confine resta la fine reale del brano
    return [int(c) for c in np.diff(kept)]

def fixed_frame_counts(total_frames, num_transitions, min_frames=4):
    edges = np.round(np.linspace(0, total_frames, num_transitions + 1)).astype(int)
    return [max(min_frames, int(c)) for c in np.diff(edges)]

def repeat_pattern_to_length(pattern, target_frames, min_frames=4):
    """Ripete il pattern ritmico fino a coprire ESATTAMENTE target_frames."""
    counts, total, j = [], 0, 0
    while total < target_frames:
        remaining = target_frames - total
        c = min(pattern[j % len(pattern)], remaining)
        if remaining - c < min_frames and remaining - c > 0:
            c = remaining  # evita un'ultima transizione troppo corta
        counts.append(c)
        total += c
        j += 1
    return counts

# ─────────────────────────────────────────────
# AUDIO: ANALISI + MUX
# ─────────────────────────────────────────────
def analyze_audio(audio_path, grid_mode):
    import librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])  # librosa >= 0.10 restituisce array
    bt = np.asarray(librosa.frames_to_time(beat_frames, sr=sr), dtype=np.float64)
    duration = len(y) / sr
    n_detected = len(bt)
    if len(bt) == 0 or bt[0] > 0.05:
        bt = np.insert(bt, 0, 0.0)
    if bt[-1] < duration - 0.05:
        bt = np.append(bt, duration)
    bt = adjust_beat_grid(bt, grid_mode)
    return bpm, bt, duration, n_detected

def get_ffmpeg_path():
    """ffmpeg incluso in imageio-ffmpeg: nessun packages.txt necessario su Streamlit Cloud."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def mux_audio(video_path, audio_path):
    muxed = video_path.replace(".mp4", "_audio.mp4")
    cmd = [get_ffmpeg_path(), "-y", "-i", video_path, "-i", audio_path,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", muxed]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(muxed):
        return muxed, None
    return None, (result.stderr or "")[-300:]

# ─────────────────────────────────────────────
# RENDER IN STREAMING
# ─────────────────────────────────────────────
def build_video(images, image_sequence, frame_counts, effect_choice, intensity, easing,
                ffmpeg_preset, fps, rnd, on_progress=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8,
                                output_params=["-preset", ffmpeg_preset])
    mixed_used, written = [], 0
    try:
        for i, n_fr in enumerate(frame_counts):
            idx1, idx2 = get_pair_indices(i, image_sequence)
            eff_name = pick_effect_for_transition(effect_choice, rnd)
            if effect_choice == MIXED_LABEL:
                mixed_used.append(eff_name)
            if on_progress:
                on_progress(i, len(frame_counts), eff_name)
            for fr in effect_funcs[eff_name](images[idx1], images[idx2], n_fr, intensity, easing):
                writer.append_data(fr)
                written += 1
    finally:
        writer.close()
    return video_path, written, mixed_used

def _safe_remove(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

# ─────────────────────────────────────────────
# REPORT BILINGUE IT / EN  (formato  "IT / EN :: valore")
# ─────────────────────────────────────────────
def _row(it, en, value):
    return f"{(it + ' / ' + en).ljust(30)} :: {value}"

def generate_report(p: dict) -> str:
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    line = "━" * 37
    rows = [
        _row("Effetto", "Effect", p["effect"]),
        _row("Easing", "Easing", p["easing"]),
        _row("Formato", "Format", p["format"]),
        _row("Risoluzione", "Resolution", p["resolution"]),
        _row("Durata", "Duration", f"{p['duration']} s"),
        _row("FPS", "FPS", p["fps"]),
        _row("Frame totali", "Total frames", p["total_frames"]),
        _row("Immagini", "Images", p["num_images"]),
        _row("Transizioni", "Transitions", p["num_transitions"]),
        _row("Intensità", "Intensity", f"{p['intensity']} / 50"),
        _row("Seed", "Seed", p["seed"]),
        _row("Preset encoding", "Encoding preset", p["ffmpeg_preset"]),
    ]
    if p.get("loop_mode") == "ping_pong":
        rows.append(_row("Loop", "Loop", "ping-pong"))
    if p.get("mixed_effects"):
        rows.append(_row("Sequenza mista", "Mixed sequence", ", ".join(p["mixed_effects"])))

    if p.get("music_enabled") and p.get("bpm"):
        audio_dur = f"{p['audio_duration']:.1f} s" if p.get("audio_duration") else "—"
        audio_rows = [
            _row("File audio", "Audio file", p["music_name"]),
            _row("BPM medio", "Avg BPM", f"{p['bpm']:.1f}"),
            _row("Beat rilevati", "Beats detected", p.get("n_beats_detected") or "—"),
            _row("Griglia beat", "Beat grid", p.get("beat_grid_label", "—")),
            _row("Durata brano", "Track length", audio_dur),
            _row("Modalità durata", "Duration mode",
                 "Tutto il brano / Full track" if p.get("full_song_mode") else "Manuale / Manual"),
        ]
    else:
        audio_rows = ["🔇 Nessun audio / No audio"]

    desc = effect_descriptions.get(p["effect"], "Effetto random diverso per ogni transizione (sequenza mista).")
    tag = "SequenzaMista" if p["effect"] == MIXED_LABEL else p["effect"].replace(" ", "")
    sync_line = (f"🎵 Beat-sync @ {p['bpm']:.1f} BPM — {p['music_name']}"
                 if p.get("music_enabled") and p.get("bpm") else "")
    stack = "Python · OpenCV · NumPy · imageio · Streamlit"
    if p.get("music_enabled"):
        stack += " · librosa · ffmpeg"

    report = f"""
╔══════════════════════════════════════════╗
║  🎞️  FRAME-TO-FRAME FX 3.1 — by Loop507  ║
╚══════════════════════════════════════════╝

Generato il / Generated :: {now}

{line}
🎬 PARAMETRI / PARAMETERS
{line}
{chr(10).join(rows)}

{line}
🎵 AUDIO
{line}
{chr(10).join(audio_rows)}

{line}
📝 DESCRIZIONE / DESCRIPTION
{line}
{desc}

{line}
📣 CAPTION SOCIAL / YOUTUBE
{line}
🎞️ {p['effect']} transition — {p['format']} | {p['duration']}s | {p['fps']}fps | seed {p['seed']}
Creato con Frame-to-Frame FX by Loop507 / Made with Frame-to-Frame FX by Loop507
{sync_line}

#VideoArt #FrameToFrame #Loop507 #VisualFX #{tag} #TransitionFX #Reels #Shorts

{line}
🛠️  STACK
{line}
{stack}

╔══════════════════════════════════════════╗
║         Loop507 — All rights reserved    ║
╚══════════════════════════════════════════╝
"""
    return report.strip()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
STATE_KEYS = ["video_bytes", "audio_bytes", "audio_name", "report_text", "video_ready"]
for key in STATE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# INTERFACCIA
# ─────────────────────────────────────────────
st.title("🎞️ Frame-to-Frame FX Video Generator 3.1 by Loop507")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Carica almeno 2 (massimo 10) immagini",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

with col2:
    effect_choice = st.selectbox("Effetto", list(effect_funcs.keys()) + [MIXED_LABEL])
    easing_choice = st.selectbox("Curva easing", ["smootherstep", "smoothstep", "sinusoidal", "linear"])
    format_choice = st.selectbox("Formato video", ["1:1", "16:9", "9:16"])

col3, col4 = st.columns(2)
with col3:
    intensity = st.slider("Intensità effetto", 1, 50, 15)
    seed_input = st.number_input("Seed (0 = casuale)", min_value=0, max_value=2**31 - 2, value=0, step=1,
                                 help="Stesso seed + stesse immagini/impostazioni = stesso risultato")
    use_music = st.toggle("🎵 Sincronizza musica ai beat", value=False)
    audio_file = None
    grid_label = list(GRID_LABELS.keys())[0]
    if use_music:
        audio_file = st.file_uploader("Carica brano (MP3 / WAV)", type=["mp3", "wav"])
        grid_label = st.radio("Griglia beat (correzione tempo dimezzato/raddoppiato)",
                              list(GRID_LABELS.keys()), horizontal=False)
with col4:
    duration_mode = st.radio(
        "Durata video",
        ["⏱️ Durata manuale", "🎵 Tutto il brano"],
        disabled=(not use_music or audio_file is None),
        help="'Tutto il brano' è disponibile solo con un brano caricato"
    )
    duration = st.slider("Durata manuale (secondi)", 1, 60, 5,
                         disabled=(duration_mode == "🎵 Tutto il brano"))
    loop_mode_label = st.radio(
        "Loop tra immagini (quando servono più transizioni delle immagini caricate)",
        ["🔁 Ciclico", "🏓 Ping-pong (avanti/indietro)"],
        horizontal=True,
    )
    loop_mode = "ping_pong" if "Ping-pong" in loop_mode_label else "cyclic"
    speed_label = st.radio(
        "Priorità render finale",
        ["⚖️ Bilanciato", "⚡ Velocità (file leggermente più pesante)", "🎯 Qualità (più lento)"],
        horizontal=True,
    )
    ffmpeg_preset = {"⚖️ Bilanciato": "medium", "⚡ Velocità (file leggermente più pesante)": "veryfast",
                     "🎯 Qualità (più lento)": "slow"}[speed_label]

# ─────────────────────────────────────────────
# ANTEPRIMA RAPIDA
# ─────────────────────────────────────────────
preview_btn = st.button("🔍 Anteprima rapida (prime 2 immagini, bassa risoluzione)")

if preview_btn:
    if not uploaded_files or len(uploaded_files) < 2:
        st.warning("⚠️ Carica almeno 2 immagini per l'anteprima.")
    else:
        with st.spinner("Genero anteprima..."):
            if seed_input:
                random.seed(int(seed_input))
                np.random.seed(int(seed_input))
            prev_size = (320, 320) if format_choice == "1:1" else ((480, 270) if format_choice == "16:9" else (270, 480))
            im1 = resize_crop_fill(np.array(Image.open(uploaded_files[0]).convert("RGB")), *prev_size)
            im2 = resize_crop_fill(np.array(Image.open(uploaded_files[1]).convert("RGB")), *prev_size)
            eff_name = pick_effect_for_transition(effect_choice, random.Random(int(seed_input) or None))
            prev_path = os.path.join(tempfile.gettempdir(), "ftf_preview.mp4")
            writer_p = imageio.get_writer(prev_path, fps=12, codec="libx264", quality=6,
                                          macro_block_size=None, output_params=["-preset", "ultrafast"])
            try:
                for fr in effect_funcs[eff_name](im1, im2, 24, intensity, easing_choice):
                    writer_p.append_data(fr)
            finally:
                writer_p.close()
            st.caption(f"Effetto usato: **{eff_name}**")
            st.video(prev_path)

# ─────────────────────────────────────────────
# GENERAZIONE
# ─────────────────────────────────────────────
video_btn = st.button("🎬 Genera Video")

if video_btn and uploaded_files and 2 <= len(uploaded_files) <= 10:

    format_dims = {"1:1": (720, 720), "16:9": (1280, 720), "9:16": (720, 1280)}
    target_size = format_dims[format_choice]
    FPS = 24

    # stato della generazione precedente azzerato (niente audio "fantasma" nei download)
    for key in STATE_KEYS:
        st.session_state[key] = None

    seed_used = int(seed_input) if seed_input else random.randrange(1, 2**31 - 1)
    random.seed(seed_used)
    np.random.seed(seed_used)
    rnd = random.Random(seed_used)

    images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
    images = [resize_crop_fill(img, *target_size) for img in images]
    image_sequence = build_image_sequence(len(images), loop_mode)
    num_transitions = len(images) - 1

    bpm_value = music_name = tmp_audio_path = audio_duration_sec = n_beats_detected = beat_times = None
    full_song_mode = (duration_mode == "🎵 Tutto il brano" and use_music and audio_file is not None)
    grid_mode = GRID_LABELS[grid_label]

    # ── Beat detection reale ───────────────────────
    if use_music and audio_file is not None:
        try:
            audio_bytes_raw = audio_file.getvalue()
            suffix = os.path.splitext(audio_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
                tmp_audio.write(audio_bytes_raw)
                tmp_audio_path = tmp_audio.name
            st.session_state["audio_bytes"] = audio_bytes_raw
            st.session_state["audio_name"] = audio_file.name
            music_name = audio_file.name

            bpm_value, beat_times, audio_duration_sec, n_beats_detected = analyze_audio(tmp_audio_path, grid_mode)
            pattern = beat_frame_counts(beat_times, FPS)
            if not pattern:
                raise ValueError("nessun intervallo di beat valido rilevato")
            st.info(f"🎵 BPM: **{bpm_value:.1f}** — {n_beats_detected} beat rilevati — "
                    f"griglia: {grid_label} — durata brano: **{audio_duration_sec:.1f}s** — "
                    f"le transizioni seguono il ritmo reale del brano.")
        except ImportError:
            st.warning("⚠️ librosa non installato. pip install librosa. Procedo senza sync.")
            beat_times, full_song_mode = None, False
        except Exception as e:
            st.warning(f"⚠️ Errore analisi audio: {e}. Procedo senza sync.")
            beat_times, full_song_mode = None, False

    # ── Frame per transizione (somma ESATTA, nessun drift) ──
    if beat_times is not None:
        pattern = beat_frame_counts(beat_times, FPS)
        if full_song_mode:
            frame_counts = pattern
        else:
            frame_counts = repeat_pattern_to_length(pattern, int(round(FPS * duration)))
    else:
        frame_counts = fixed_frame_counts(int(round(FPS * duration)), num_transitions)
    total_transitions_needed = len(frame_counts)

    progress = st.progress(0.0)
    status = st.empty()

    def _on_progress(i, n, eff):
        status.text(f"Elaborazione transizione {i + 1}/{n} — {eff}...")
        progress.progress(i / n)

    video_path, total_frames_written, mixed_effects_used = build_video(
        images, image_sequence, frame_counts, effect_choice, intensity, easing_choice,
        ffmpeg_preset, FPS, rnd, _on_progress)
    progress.progress(1.0)
    status.text("Scrittura video completata...")

    # ── Mux audio ───────────────────────────────────
    final_video_path, muxed_path = video_path, None
    if tmp_audio_path and os.path.exists(tmp_audio_path):
        try:
            muxed_path, err = mux_audio(video_path, tmp_audio_path)
            if muxed_path:
                final_video_path = muxed_path
                status.text("✅ Audio aggiunto al video!")
            else:
                st.warning(f"⚠️ ffmpeg: errore nel mux audio. Video senza audio. {err}")
        except Exception as e:
            st.warning(f"⚠️ Impossibile aggiungere audio: {e}")

    with open(final_video_path, "rb") as f:
        st.session_state["video_bytes"] = f.read()
    _safe_remove(video_path, muxed_path, tmp_audio_path)

    w, h = target_size
    report_params = {
        "effect": effect_choice, "easing": easing_choice, "format": format_choice,
        "resolution": f"{w}×{h}px", "duration": f"{total_frames_written / FPS:.1f}", "fps": FPS,
        "total_frames": total_frames_written, "num_images": len(images),
        "num_transitions": total_transitions_needed, "intensity": intensity, "seed": seed_used,
        "music_enabled": use_music and audio_file is not None, "bpm": bpm_value,
        "music_name": music_name, "full_song_mode": full_song_mode,
        "audio_duration": audio_duration_sec, "n_beats_detected": n_beats_detected,
        "beat_grid_label": grid_label, "loop_mode": loop_mode, "ffmpeg_preset": ffmpeg_preset,
        "mixed_effects": sorted(set(mixed_effects_used)) if mixed_effects_used else None,
    }
    st.session_state["report_text"] = generate_report(report_params)
    st.session_state["video_ready"] = True

    status.empty()
    progress.empty()
    st.success(f"✅ Video generato con successo! (seed {seed_used})")

elif video_btn:
    st.warning("⚠️ Carica tra 2 e 10 immagini prima di generare.")

# ─────────────────────────────────────────────
# DOWNLOAD — sempre visibili dopo la generazione (FUORI da st.button → non si resettano al rerun)
# ─────────────────────────────────────────────
if st.session_state.get("video_ready"):
    st.divider()
    has_audio = st.session_state.get("audio_bytes") is not None
    # numero di colonne SEMPRE fisso (3): evita il crash "removeChild" del frontend Streamlit
    dl_cols = st.columns(3)

    with dl_cols[0]:
        st.download_button(
            label="📥 Scarica Video (.mp4)",
            data=st.session_state["video_bytes"],
            file_name="loop507_output.mp4",
            mime="video/mp4",
            use_container_width=True,
            key="dl_video_btn",
        )

    with dl_cols[1]:
        st.download_button(
            label="📄 Scarica Report (.txt)",
            data=st.session_state["report_text"],
            file_name="loop507_report.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_report_btn",
        )

    with dl_cols[2]:
        if has_audio:
            audio_ext = os.path.splitext(st.session_state["audio_name"])[1]
            st.download_button(
                label=f"🎵 Scarica Audio ({audio_ext})",
                data=st.session_state["audio_bytes"],
                file_name=st.session_state["audio_name"],
                mime="audio/mpeg" if audio_ext == ".mp3" else "audio/wav",
                use_container_width=True,
                key="dl_audio_btn",
            )

    with st.expander("👁️ Anteprima report"):
        st.code(st.session_state["report_text"], language=None)
else:
    st.info("Carica almeno 2 immagini e premi 'Genera Video'.")
