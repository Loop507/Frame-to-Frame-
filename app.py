import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import imageio.v2 as imageio  # API legacy get_writer: la v3 la dichiara obsoleta
import os
import tempfile
import random
import subprocess

st.set_page_config(page_title="🎞️ Frame-to-Frame FX Video Generator 3.1 by Loop507", layout="wide")

# Limiti di sicurezza per la RAM (Streamlit Community Cloud: ~1 GB garantiti, fino a 2.7 GB).
# Il vincolo vero sono i MB caricati (Streamlit li tiene tutti in memoria), non il numero di foto.
MAX_IMAGES = 40
MAX_UPLOAD_MB = 150

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
# AUDIO FEATURES — una riga per frame video (allineate al brano, 0..1)
# ─────────────────────────────────────────────
N_BANDS = 32

def _smooth_ad(x, attack, decay):
    """Inseguitore attacco/rilascio lungo l'asse 0 (1D o 2D): sale in fretta, scende piano."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    v = np.zeros(x.shape[1:])
    for i in range(len(x)):
        v = v + (x[i] - v) * np.where(x[i] > v, attack, decay)
        out[i] = v
    return out

class AudioFeatures:
    """rms, onset, kick, bass, mid, high: array (n,)   bands: array (n, N_BANDS)   — tutti in 0..1
    onset = qualsiasi colpo (kick, rullante, hat);  kick = solo colpi sulle frequenze basse"""
    def __init__(self, rms, onset, bass, mid, high, bands, kick=None):
        self.rms, self.onset, self.bass = rms, onset, bass
        self.mid, self.high, self.bands = mid, high, bands
        self.kick = onset if kick is None else kick

    def __len__(self):
        return len(self.rms)

    def slice(self, start, n):
        def cut(a):
            seg = a[start:start + n]
            if len(seg) < n:  # oltre la fine del brano → silenzio
                seg = np.pad(seg, [(0, n - len(seg))] + [(0, 0)] * (a.ndim - 1))
            return seg
        return AudioFeatures(cut(self.rms), cut(self.onset), cut(self.bass),
                             cut(self.mid), cut(self.high), cut(self.bands), kick=cut(self.kick))

    @classmethod
    def synthetic(cls, n, fps=24, rng=None):
        """Ritmo finto a 120 BPM (kick + rumore liscio): gli effetti audio-reattivi
        restano vivi anche senza brano (anteprima, modalità senza musica)."""
        rng = rng or random.Random(0)
        g = np.random.default_rng(rng.randrange(2**31))
        n = max(1, int(n))
        onset = np.zeros(n)
        onset[::max(4, int(round(fps * 0.5)))] = 1.0
        onset = np.clip(_smooth_ad(onset, 1.0, 0.35), 0, 1)
        noise = lambda: np.clip(_smooth_ad(g.random(n), 0.3, 0.3) * 1.6 - 0.2, 0, 1)
        bass = np.clip(0.15 + 0.85 * onset, 0, 1)
        rms = np.clip(0.25 + 0.5 * onset + 0.25 * noise(), 0, 1)
        bands = np.clip(_smooth_ad(g.random((n, N_BANDS)), 0.4, 0.3) * 1.2 - 0.1 + onset[:, None] * 0.4, 0, 1)
        return cls(rms, onset, bass, noise(), noise(), bands)

def _au(audio, n):
    if audio is None or len(audio) < n:
        return AudioFeatures.synthetic(n, 24, random.Random(random.randrange(2**31)))
    return audio

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


# ─────────────────────────────────────────────
# EFFETTI AUDIO-REATTIVI — firma: (..., alphas=None, audio=None)
# 'audio' è la fetta di AudioFeatures dei frame di QUESTA transizione.
# ─────────────────────────────────────────────
def _flow_full(img1, img2):
    """Optical flow Farneback img1→img2 a metà risoluzione, riportato a piena (px)."""
    h, w = img1.shape[:2]
    scale = 0.5 if max(h, w) > 480 else 1.0
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    g1 = cv2.cvtColor(cv2.resize(img1, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    flow = cv2.GaussianBlur(flow, (0, 0), 3)
    return (cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale).astype(np.float32)

def _chroma_split(frame, shift):
    h, w = frame.shape[:2]
    r, g, b = cv2.split(frame)
    r = cv2.warpAffine(r, np.float32([[1, 0, shift], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(b, np.float32([[1, 0, -shift], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([r, g, b])

def spectrum_reveal_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """img2 emerge dal basso attraverso un equalizzatore: ogni barra segue la SUA banda dello spettro,
    con tappi bianchi sui picchi e spaziature tra le barre che compaiono a metà transizione."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    nbars = 48
    bar_w = w / nbars
    bar_idx = np.minimum((np.arange(w) // bar_w).astype(int), nbars - 1)
    gap = ((np.arange(w) % bar_w) / bar_w) > 0.88
    y_norm = ((h - 1 - np.arange(h)) / h).astype(np.float32)[:, None]
    band_x = np.linspace(0, N_BANDS - 1, nbars)
    spread = 0.35 + intensity / 40.0
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    for k, a in enumerate(al):
        env = _env(a)
        e_bars = np.interp(band_x, np.arange(N_BANDS), au.bands[k])
        f_bar = np.clip(a + spread * env * (e_bars - 0.5) * 1.5, 0, 1)
        f_col = f_bar[bar_idx].astype(np.float32)[None, :]
        mask = np.clip((f_col - y_norm) * (h / 8.0) + 0.5, 0, 1)
        mask[:, gap] *= (1.0 - env ** 2)
        m3 = mask[:, :, None]
        out = f1 * (1 - m3) + f2 * m3
        cap = (np.abs(f_col - y_norm) * h < 2.5).astype(np.float32)[:, :, None] * (env * (0.4 + 0.6 * float(au.rms[k])))
        out = out * (1 - cap) + 255.0 * cap
        yield np.clip(out, 0, 255).astype(np.uint8)

def kick_zoom_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Evoluzione di Zoom Random: ad ogni colpo (onset) il fuoco SALTA in un punto nuovo, lo zoom
    scatta, l'immagine ruota di un soffio e i canali RGB si separano. Il basso fa tremare l'inquadratura."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    amp = intensity / 50.0
    cx, cy, direction, prev_kick = w / 2.0, h / 2.0, 1, 0.0
    for k, a in enumerate(al):
        env = _env(a)
        kick, bass = float(au.kick[k]), float(au.bass[k])
        if kick > 0.55 and prev_kick <= 0.55:
            cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
            direction = random.choice((-1, 1))
        prev_kick = kick
        zoom = 1.0 + amp * env * (0.4 + 1.0 * kick)
        M = cv2.getRotationMatrix2D((cx, cy), direction * kick * env * amp * 4.0, zoom)
        M[0, 2] += random.uniform(-1, 1) * bass * env * amp * 40
        M[1, 2] += random.uniform(-1, 1) * bass * env * amp * 40
        z1 = cv2.warpAffine(img1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        z2 = cv2.warpAffine(img2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        yield _chroma_split(_blend(z1, z2, a), kick * env * amp * 30.0)

def flow_smear_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Datamosh a vettori di moto: i pixel di img1 vengono TRASCINATI lungo l'optical flow verso img2
    in modo ricorsivo (nessun crossfade). Kick e volume spingono il movimento; img2 emerge prima
    dove c'è poco moto e per ultima dove il moto è forte."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    flow = _flow_full(img1, img2)
    mag = np.linalg.norm(flow, axis=2)
    mag_n = np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)
    thr = (0.15 + 0.7 * (1.0 - mag_n)).astype(np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    f2 = img2.astype(np.float32)
    cur = img1.copy()
    amp = intensity / 15.0
    for k, a in enumerate(al):
        env = _env(a)
        g = (0.05 + 0.30 * float(au.onset[k]) + 0.12 * float(au.rms[k])) * amp * (env ** 0.5)
        g = min(g, 0.7)
        cur = cv2.remap(cur, xs - g * flow[..., 0], ys - g * flow[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wmap = np.clip((a * 1.25 - thr) / 0.25, 0, 1)[:, :, None]
        yield (cur.astype(np.float32) * (1 - wmap) + f2 * wmap).astype(np.uint8)

def block_shuffle_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Macroblocchi corrotti: la transizione avviene a blocchi (ognuno con la sua soglia), e ad ogni
    colpo una raffica di blocchi viene spostata e, sui colpi forti, invertita. Il basso allarga lo scarto."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    bs = int(np.clip(80 - intensity * 1.2, 16, 64))
    nby, nbx = -(-h // bs), -(-w // bs)
    tau = np.random.uniform(0, 0.9, (nby, nbx))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    up = lambda m: np.repeat(np.repeat(m, bs, axis=0), bs, axis=1)[:h, :w]
    for k, a in enumerate(al):
        env = _env(a)
        kick, bass, rms = float(au.onset[k]), float(au.bass[k]), float(au.rms[k])
        base = np.where(up(a > tau)[:, :, None], img2, img1)
        p = float(np.clip(0.03 + 0.85 * kick + 0.25 * rms, 0, 1)) * env
        M = np.random.random((nby, nbx)) < p
        dx = np.where(M, np.random.randint(-2, 3, (nby, nbx)) * bs * (0.5 + bass), 0).astype(np.float32)
        dy = np.where(M, np.random.randint(-1, 2, (nby, nbx)) * bs * 0.5 * (0.5 + bass), 0).astype(np.float32)
        out = cv2.remap(base, xs - up(dx), ys - up(dy), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        if kick > 0.6:
            inv = up(M & (np.random.random((nby, nbx)) < 0.25 * (kick - 0.4)))
            out[inv] = 255 - out[inv]
        yield out

def slit_scan_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Fronte d'onda ondulato che spazza l'immagine: la forma dell'onda si muove con le medie
    (fase) e si gonfia con i bassi (ampiezza); lungo il fronte i pixel vengono tirati
    orizzontalmente in proporzione al volume — effetto slit-scan."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    xn = (np.arange(w) / w).astype(np.float32)[None, :]
    yn = (np.arange(h) / h).astype(np.float32)[:, None]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    freq, soft, phase = 2.0 + intensity / 12.0, 0.22, 0.0
    for k, a in enumerate(al):
        bass, mid, rms = float(au.bass[k]), float(au.mid[k]), float(au.rms[k])
        phase += 0.12 + 0.8 * mid
        amp_w = min(0.45, 0.30 * (0.35 + bass))
        q = np.clip(0.15 + 0.7 * xn + amp_w * np.sin(2 * np.pi * freq * yn + phase), 0, 1)
        m = np.clip((a - q * (1 - soft)) / soft, 0, 1)
        out = (f1 * (1 - m[:, :, None]) + f2 * m[:, :, None]).astype(np.uint8)
        edge = 4.0 * m * (1.0 - m)
        disp = intensity * 1.5 * (0.3 + rms) * np.sin(2 * np.pi * yn * freq * 3 - phase * 2)
        yield cv2.remap(out, xs + edge * disp, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def feedback_tunnel_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Video feedback: ogni frame rientra nel successivo, ingrandito e ruotato, con deriva di tinta.
    Il volume decide quanta traccia resta, i bassi la rotazione, gli alti la velocità del colore;
    ogni colpo accende un flash."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    amp = intensity / 15.0
    prev = None
    for k, a in enumerate(al):
        env = _env(a)
        rms, bass, high, kick = float(au.rms[k]), float(au.bass[k]), float(au.high[k]), float(au.kick[k])
        cur = _blend(img1, img2, a)
        if prev is None:
            out = cur
        else:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), (bass - 0.5) * 3.0 * amp, 1.0 + 0.02 + 0.05 * rms * amp)
            pt = cv2.warpAffine(prev, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            hsv = cv2.cvtColor(pt, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(3 + 14 * high)) % 180).astype(np.uint8)
            pt = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            fb = float(np.clip(0.55 + 0.4 * rms, 0, 0.93)) * env
            out = cv2.addWeighted(cur, 1 - fb, pt, fb, 0)
        flash = int(kick * intensity * 1.5 * env)
        if flash > 0:
            out = cv2.add(out, (flash, flash, flash, 0))
        prev = out
        yield out

def pixel_sort_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Pixel sorting a blocchi: righe a caso vengono ordinate per luminosità in segmenti. I colpi
    (kick) decidono QUANTE righe, i bassi la lunghezza dei segmenti (16→256 px), e ogni colpo
    forte inverte la direzione (orizzontale ↔ verticale)."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    lum_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    amp = min(1.0, intensity / 20.0 + 0.3)
    vertical, prev = False, 0.0
    for k, a in enumerate(al):
        env = _env(a)
        kick, bass = float(au.kick[k]), float(au.bass[k])
        if kick > 0.55 and prev <= 0.55:
            vertical = not vertical
        prev = kick
        frame = _blend(img1, img2, a)
        p = float(np.clip((0.1 + 0.9 * kick) * env * amp, 0, 1))
        L = 2 ** int(4 + 4 * min(0.999, bass))
        work = np.ascontiguousarray(frame.transpose(1, 0, 2)) if vertical else frame
        Hh, Ww = work.shape[:2]
        n = Ww // L
        idx = np.where(np.random.random(Hh) < p)[0]
        if n >= 1 and len(idx) > 0:
            sub = work[idx, :n * L]
            lum = (sub.astype(np.float32) @ lum_w).reshape(len(idx), n, L)
            order = np.argsort(lum, axis=2)
            chunks = sub.reshape(len(idx), n, L, 3)
            work[idx, :n * L] = np.take_along_axis(chunks, order[..., None], axis=2).reshape(len(idx), n * L, 3)
        yield np.ascontiguousarray(work.transpose(1, 0, 2)) if vertical else work

def shockwave_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Onde d'urto: ogni kick fa nascere un anello che si espande da un punto a caso e deforma
    l'immagine come acqua; i canali RGB si rifrangono in modo diverso (aberrazione). Fino a 4 anelli."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    size = float(max(h, w))
    speed, width = 0.04 * size, 0.09 * size
    rings, prev = [], 0.0
    for k, a in enumerate(al):
        env = _env(a)
        kick = float(au.kick[k])
        if kick > 0.5 and prev <= 0.5:
            cx, cy = random.uniform(0.25 * w, 0.75 * w), random.uniform(0.25 * h, 0.75 * h)
            dx, dy = xs - cx, ys - cy
            dist = np.sqrt(dx * dx + dy * dy) + 1e-3
            rings = (rings + [{"dx": dx / dist, "dy": dy / dist, "dist": dist, "age": 0, "amp": kick}])[-4:]
        prev = kick
        frame = _blend(img1, img2, a)
        ox, oy = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        for r in rings:
            r["age"] += 1
            u = (r["dist"] - r["age"] * speed) / width
            prof = np.exp(-u * u) * np.sin(u * 3.0)
            amp_px = float(r["amp"] * intensity * 3.4 * np.exp(-r["age"] * 0.10) * env)
            ox += amp_px * prof * r["dx"]
            oy += amp_px * prof * r["dy"]
        rings = [r for r in rings if r["age"] < 40]
        chans = [cv2.remap(c, xs - f * ox, ys - f * oy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                 for c, f in zip(cv2.split(frame), (1.35, 1.0, 0.65))]
        yield cv2.merge(chans)

def kaleido_pulse_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Caleidoscopio a impulsi: lo specchio si accende a metà transizione, il numero di spicchi
    cambia a ogni colpo (3–10), i medi lo fanno ruotare e i bassi lo dilatano."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    cx, cy = w / 2.0, h / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = xs - cx, ys - cy
    r, th = np.sqrt(dx * dx + dy * dy), np.arctan2(dy, dx)
    n_seg, rot, prev = 6, 0.0, 0.0
    for k, a in enumerate(al):
        env = _env(a)
        kick = float(au.kick[k])
        if kick > 0.55 and prev <= 0.55:
            n_seg = random.choice((3, 4, 5, 6, 8, 10))
        prev = kick
        rot += 0.01 + 0.08 * float(au.mid[k])
        seg = float(2 * np.pi / n_seg)
        th_f = np.abs(((th + rot) % seg) - seg / 2.0)
        rs = r * (1.0 + 0.25 * float(au.bass[k]))
        frame = _blend(img1, img2, a)
        kal = cv2.remap(frame, (cx + rs * np.cos(th_f)).astype(np.float32),
                        (cy + rs * np.sin(th_f)).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = float(np.clip(env ** 0.7 * (0.55 + 0.45 * kick), 0, 1))
        yield cv2.addWeighted(frame, 1 - m, kal, m, 0)

def time_slice_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Fette temporali: l'immagine è divisa in 32 fasce orizzontali (bassi in basso, acuti in alto);
    ogni fascia pesca da un frame PASSATO tanto più vecchio quanto più forte è la sua banda di
    frequenza, e scivola di lato. Lo spettro diventa letteralmente tempo."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    nb = 32
    bounds = [(b * h // nb, (b + 1) * h // nb) for b in range(nb)]
    hist = []
    for k, a in enumerate(al):
        env = _env(a)
        hist = (hist + [_blend(img1, img2, a)])[-14:]
        maxd = min(len(hist) - 1, int(2 + intensity / 4))
        delays = np.round(np.clip(au.bands[k][::-1] * 1.2, 0, 1) * maxd * env).astype(int)
        out = np.empty_like(hist[-1])
        for b, (r0, r1) in enumerate(bounds):
            d = int(delays[b])
            src = hist[-1 - d][r0:r1]
            shift = int(d * intensity * 0.6) * (1 if b % 2 else -1)
            out[r0:r1] = np.roll(src, shift, axis=1) if shift else src
        yield out

def shatter_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Frantumazione: l'immagine si spezza in 36 schegge (Voronoi). Ogni scheggia passa a img2 al suo
    momento; a ogni kick le schegge vengono SPARATE verso l'esterno e tornano al loro posto, mentre
    le crepe si scuriscono in proporzione agli acuti."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    n = 36
    seeds = np.random.rand(n, 2) * np.array([w, h])
    sw, sh = max(8, w // 4), max(8, h // 4)
    gx, gy = np.meshgrid(np.arange(sw) * 4.0, np.arange(sh) * 4.0)
    d2 = (gx[..., None] - seeds[:, 0]) ** 2 + (gy[..., None] - seeds[:, 1]) ** 2
    label = cv2.resize(np.argmin(d2, axis=2).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    edges = ((label != np.roll(label, 1, 0)) | (label != np.roll(label, 1, 1))).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(bool)
    tau = np.random.uniform(0, 0.85, n)
    vec = seeds - np.array([w / 2.0, h / 2.0])
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6) + np.random.uniform(-0.4, 0.4, (n, 2))
    scale = np.random.uniform(0.5, 1.5, n)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    energy = 0.0
    for k, a in enumerate(al):
        env = _env(a)
        kick, bass, high = float(au.kick[k]), float(au.bass[k]), float(au.high[k])
        energy = max(energy * 0.8, kick)
        base = np.where((a > tau)[label][:, :, None], img2, img1)
        mag = intensity * 1.8 * energy * env * (0.4 + bass)
        dxc = (vec[:, 0] * scale * mag).astype(np.float32)
        dyc = (vec[:, 1] * scale * mag).astype(np.float32)
        out = cv2.remap(base, xs - dxc[label], ys - dyc[label], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ls = float(np.clip(env * (0.25 + 0.75 * high), 0, 1)) * 0.85
        out[edges] = (out[edges] * (1 - ls)).astype(np.uint8)
        yield out

def luma_crush_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Luma-key brutalista: img2 sostituisce img1 a partire dalle zone scure, con bordo dithering
    Bayer. Sui colpi l'immagine viene POSTERIZZATA fino a 2 livelli, i canali si permutano
    (colori falsi) e gli acuti aggiungono rumore."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    bayer = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                      [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                      [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                      [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) / 64.0
    tile = np.tile(bayer, (h // 8 + 1, w // 8 + 1))[:h, :w]
    lum1 = (img1.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    key = np.clip(lum1 * 0.85 + 0.15 * tile, 0, 1)
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    perms = [(2, 0, 1), (1, 2, 0), (2, 1, 0), (0, 2, 1)]
    perm, prev, ws = None, 0.0, 0.2
    for k, a in enumerate(al):
        env = _env(a)
        kick, high = float(au.kick[k]), float(au.high[k])
        m = np.clip((a - key * (1 - ws)) / ws, 0, 1)[:, :, None]
        frame = f1 * (1 - m) + f2 * m
        crush = float(np.clip(env * (0.25 + 0.75 * kick), 0, 1))
        q = 256.0 / max(2, int(round(12 - 10 * crush)))
        crushed = np.floor(frame / q) * q + q / 2
        if kick > 0.6 and prev <= 0.6:
            perm = random.choice(perms)
        elif kick <= 0.45:
            perm = None
        prev = kick
        if perm is not None:
            crushed = crushed[..., list(perm)]
        crushed = crushed + (np.random.random((h, w, 1)).astype(np.float32) - 0.5) * 60.0 * high * env
        mix = float(np.clip(env * (0.3 + 0.7 * kick), 0, 1))
        yield np.clip(frame * (1 - mix) + crushed * mix, 0, 255).astype(np.uint8)

AUDIO_EFFECTS = {"Spectrum Reveal", "Kick Zoom", "Flow Smear", "Block Shuffle", "Slit Scan", "Feedback Tunnel",
                 "Pixel Sort", "Shockwave", "Kaleido Pulse", "Time Slice", "Shatter", "Luma Crush"}

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
    "Spectrum Reveal": spectrum_reveal_effect,
    "Kick Zoom": kick_zoom_effect,
    "Flow Smear": flow_smear_effect,
    "Block Shuffle": block_shuffle_effect,
    "Slit Scan": slit_scan_effect,
    "Feedback Tunnel": feedback_tunnel_effect,
    "Pixel Sort": pixel_sort_effect,
    "Shockwave": shockwave_effect,
    "Kaleido Pulse": kaleido_pulse_effect,
    "Time Slice": time_slice_effect,
    "Shatter": shatter_effect,
    "Luma Crush": luma_crush_effect,
    "Combo Creative 1": combo_effect_preset1,
    "Combo Creative 2": combo_effect_preset2,
    "Combo Creative 3": combo_effect_preset3,
}

effect_descriptions = {
    "Fade": "Classic dissolve between one frame and the next.",
    "Morph": "Optical-flow morph (Farneback): pixels slide from one image to the other, then crossfade. Pure DSP.",
    "Glitch": "Digital artifacts and row displacement typical of a corrupted signal.",
    "Pixelate": "Progressive pixelation with random scale for an 8-bit look.",
    "Corrupt Lines": "Corrupted data lines randomly appearing on the frame.",
    "Wave Distort": "Sinusoidal wave distortion of the horizontal rows.",
    "Zoom": "Punch-in from the center, returns to 1x at the end of the transition.",
    "Zoom Random": "Zoom from a random focus for a chaotic effect, retracts at the end of the transition.",
    "Wipe Direzionale": "Horizontal left-to-right wipe with a soft edge.",
    "Wipe Diagonale": "Diagonal wipe from top-left to bottom-right corner with a soft edge.",
    "Iris Circolare": "Circular opening from the center to the edges, cinema iris style.",
    "Color Echo": "Chromatic aberration: R and B channels split in opposite directions.",
    "Particle Float": "White floating particles over the blend.",
    "Film Look": "Cinematic grain with exposure flicker.",
    "VHS": "Magnetic tape artifacts with red-channel equalization.",
    "Motion Blur": "Horizontal motion blur on the blended frame.",
    "Scanlines": "CRT-monitor style scanlines.",
    "Burn-in": "Line overlay with a phosphor burn-in effect.",
    "Light Leaks": "Colored light halos projected on the frame, analog style.",
    "Spectrum Reveal": "Audio-reactive: the next image rises through an equalizer, each bar following its own frequency band.",
    "Kick Zoom": "Audio-reactive: the focus jumps on every hit, zoom and rotation snap on the kick, bass shakes the frame.",
    "Flow Smear": "Audio-reactive datamosh: pixels are dragged along optical-flow vectors, pushed by kick and volume.",
    "Block Shuffle": "Audio-reactive macroblock corruption: blocks displace and invert on every hit, bass widens the offset.",
    "Slit Scan": "Audio-reactive wavefront sweep: mids move the wave, bass swells it, volume pulls pixels along the front.",
    "Feedback Tunnel": "Audio-reactive video feedback: zoom, rotation and hue drift recirculate, kick triggers a flash.",
    "Pixel Sort": "Audio-reactive pixel sorting: kick sets how many rows get sorted, bass the segment length, strong hits flip the direction.",
    "Shockwave": "Audio-reactive shockwaves: every kick spawns an expanding ring that bends the image like water, with RGB refraction.",
    "Kaleido Pulse": "Audio-reactive kaleidoscope: the segment count changes on every hit, mids rotate it, bass dilates it.",
    "Time Slice": "Audio-reactive time slicing: each of 32 bands pulls from a past frame as old as its frequency energy.",
    "Shatter": "Audio-reactive shatter: the image breaks into Voronoi shards that are blasted outward on every kick.",
    "Luma Crush": "Audio-reactive luma-key with Bayer dithering: hits posterize the image, swap color channels and add noise.",
    "Combo Creative 1": "Glitch + Pixelate in sequence (continuous alpha curve).",
    "Combo Creative 2": "Wave Distort + Color Echo + Particle Float in sequence.",
    "Combo Creative 3": "Motion Blur + Film Look + Zoom Random in sequence.",
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

def load_image_fit(file, target_w, target_h):
    """Apre UNA foto, la decodifica già ridotta (JPEG draft 1/2, 1/4, 1/8), applica l'orientamento EXIF
    e la porta al formato scelto. Il picco di RAM è quello di una sola foto, non di tutte insieme."""
    with Image.open(file) as im:
        side = max(target_w, target_h)          # quadrato: sicuro anche per foto ruotate dall'EXIF
        im.draft("RGB", (side, side))
        im = ImageOps.exif_transpose(im)
        arr = np.array(im.convert("RGB"))
    return resize_crop_fill(arr, target_w, target_h)

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

GRID_EN = {
    "normal": "×1 (detected beats)",
    "half": "×½ (one transition every 2 beats)",
    "double": "×2 (split each beat)",
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
def load_audio_streaming(path, target_sr=22050, block_seconds=30):
    """Decodifica a blocchi (mono, 22.05 kHz): la RAM è ~0.09 MB per secondo di brano, mentre
    librosa.load tiene l'intero file decodificato stereo a piena frequenza (~0.7 MB/s in transitorio)."""
    try:
        import soundfile as sf
        import soxr
        with sf.SoundFile(path) as f:
            sr_in = f.samplerate
            rs = soxr.ResampleStream(sr_in, target_sr, 1, dtype="float32") if sr_in != target_sr else None
            out = []
            for block in f.blocks(blocksize=int(block_seconds * sr_in), dtype="float32", always_2d=True):
                mono = np.ascontiguousarray(block.mean(axis=1), dtype=np.float32)
                out.append(rs.resample_chunk(mono) if rs else mono)
            if rs:
                out.append(rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
        return np.concatenate(out), target_sr
    except Exception:
        import librosa  # formati/versioni non gestiti da soundfile: percorso classico
        return librosa.load(path, sr=target_sr, mono=True)

def _spectral_pass(y, sr, hop=512, seg_seconds=60):
    """STFT a blocchi da ~60 s → rms, mel a 128 bande (beat) e mel a 32 bande (feature).
    La RAM resta piatta qualunque sia la durata del brano (a blocco unico cresceva ~1 MB/s di audio)."""
    import librosa
    seg = (int(seg_seconds * sr) // hop) * hop
    ctx = 2048  # campioni di contesto ai lati di ogni blocco: i frame ai bordi restano identici a quelli continui
    rms_l, m128_l, m32_l = [], [], []
    for start in range(0, len(y), seg):
        lo, hi = max(0, start - ctx), min(len(y), start + seg + ctx)
        S = np.abs(librosa.stft(y[lo:hi], n_fft=2048, hop_length=hop))
        skip = (start - lo) // hop  # frame di contesto da scartare a sinistra
        S = S[:, skip:] if start + seg >= len(y) else S[:, skip:skip + seg // hop]
        P = S ** 2
        rms_l.append(librosa.feature.rms(S=S, frame_length=2048, hop_length=hop)[0])
        m128_l.append(librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=sr, n_mels=128), top_db=None))
        m32_l.append(librosa.power_to_db(
            librosa.feature.melspectrogram(S=P, sr=sr, n_mels=N_BANDS, fmax=min(16000.0, sr / 2.0)), top_db=None))
    m128, m32 = np.concatenate(m128_l, axis=1), np.concatenate(m32_l, axis=1)
    return np.concatenate(rms_l), np.maximum(m128, m128.max() - 80.0), np.maximum(m32, m32.max() - 60.0)

def extract_audio_features(rms, mel_db, sr, hop, duration, fps, n_bands=N_BANDS):
    """RMS, onset, kick, bassi/medi/alti e spettro a 32 bande, normalizzati 0..1 e ricampionati
    sull'asse dei frame video (centro di ogni frame) con attacco rapido / rilascio lento."""
    import librosa
    T = mel_db.shape[1]
    t_stft = np.arange(T) * hop / sr
    n_video = int(np.ceil(duration * fps)) + 2
    t_video = (np.arange(n_video) + 0.5) / fps

    def norm(x, lo=5, hi=99):
        a, b = np.percentile(x, lo), np.percentile(x, hi)
        return np.clip((x - a) / (b - a + 1e-9), 0, 1)

    def to_video(x):
        return np.interp(t_video, t_stft, x)

    def fit(x):
        return np.pad(x, (0, max(0, T - len(x))))[:T]

    rms = norm(rms)
    bands = np.stack([norm(mel_db[b]) for b in range(n_bands)], axis=1)
    onset = fit(librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop))
    q = n_bands // 8
    kick = fit(librosa.onset.onset_strength(S=mel_db[:q], sr=sr, hop_length=hop))  # solo bassi → colpi di cassa
    bass_t, mid_t, high_t = bands[:, :q].mean(1), bands[:, q:n_bands // 2].mean(1), bands[:, n_bands // 2:].mean(1)
    bands_v = np.stack([to_video(bands[:, b]) for b in range(n_bands)], axis=1)
    return AudioFeatures(
        rms=_smooth_ad(to_video(rms), 0.6, 0.25),
        onset=np.clip(_smooth_ad(to_video(norm(onset)), 0.9, 0.35), 0, 1),
        kick=np.clip(_smooth_ad(to_video(norm(kick)), 0.9, 0.4), 0, 1),
        bass=_smooth_ad(to_video(norm(bass_t)), 0.6, 0.3),
        mid=_smooth_ad(to_video(norm(mid_t)), 0.6, 0.3),
        high=_smooth_ad(to_video(norm(high_t)), 0.6, 0.3),
        bands=np.clip(_smooth_ad(bands_v, 0.7, 0.3), 0, 1),
    )

def analyze_audio(audio_path, grid_mode, fps=24):
    import librosa
    hop = 512
    y, sr = load_audio_streaming(audio_path, 22050)
    duration = len(y) / sr
    rms, mel128, mel32 = _spectral_pass(y, sr, hop)
    del y  # il segnale non serve più: da qui lavoriamo su matrici piccole
    # stessa envelope che beat_track(y=...) calcolerebbe internamente (mel 128, mediana)
    onset_env = librosa.onset.onset_strength(S=mel128, sr=sr, hop_length=hop, aggregate=np.median)
    max_tempo_frames = int(300 * sr / hop)  # 5 minuti
    if len(onset_env) > max_tempo_frames:
        mid = len(onset_env) // 2
        excerpt = onset_env[mid - max_tempo_frames // 2: mid + max_tempo_frames // 2]
        bpm_est = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=excerpt, sr=sr, hop_length=hop))[0])
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, bpm=bpm_est)
    else:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])  # librosa >= 0.10 restituisce array
    bt = np.asarray(librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop), dtype=np.float64)
    n_detected = len(bt)
    if len(bt) == 0 or bt[0] > 0.05:
        bt = np.insert(bt, 0, 0.0)
    if bt[-1] < duration - 0.05:
        bt = np.append(bt, duration)
    bt = adjust_beat_grid(bt, grid_mode)
    try:
        features = extract_audio_features(rms, mel32, sr, hop, duration, fps)
    except Exception:
        features = None  # il beat-sync funziona comunque; gli effetti audio usano il ritmo sintetico
    return bpm, bt, duration, n_detected, features

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

def make_preview_proxy(video_path, max_side=480):
    """Copia a bassa risoluzione e bitrate del video finale (audio incluso), solo per guardarlo nel
    browser prima di scaricarlo. Il file da scaricare resta in piena qualità."""
    out = video_path.replace(".mp4", "_proxy.mp4")
    vf = (f"scale=w={max_side}:h={max_side}:force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd = [get_ffmpeg_path(), "-y", "-i", video_path, "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(out):
            with open(out, "rb") as f:
                return f.read()
    except Exception:
        pass
    finally:
        _safe_remove(out)
    return None

# ─────────────────────────────────────────────
# RENDER IN STREAMING
# ─────────────────────────────────────────────
def build_video(images, image_sequence, frame_counts, effect_choice, intensity, easing,
                ffmpeg_preset, fps, rnd, on_progress=None, features=None, quality=8, macro_block_size=16):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=quality,
                                macro_block_size=macro_block_size, output_params=["-preset", ffmpeg_preset])
    mixed_used, written, cursor = [], 0, 0
    if features is None:  # senza brano: ritmo sintetico, ma deterministico (seed)
        features = AudioFeatures.synthetic(sum(frame_counts), fps, rnd)
    try:
        for i, n_fr in enumerate(frame_counts):
            idx1, idx2 = get_pair_indices(i, image_sequence)
            eff_name = pick_effect_for_transition(effect_choice, rnd)
            if effect_choice == MIXED_LABEL:
                mixed_used.append(eff_name)
            if on_progress:
                on_progress(i, len(frame_counts), eff_name)
            kwargs = {"audio": features.slice(cursor, n_fr)} if eff_name in AUDIO_EFFECTS else {}
            cursor += n_fr
            for fr in effect_funcs[eff_name](images[idx1], images[idx2], n_fr, intensity, easing, **kwargs):
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
# REPORT  (formato  "Label :: value", numerato per sessione)
# ─────────────────────────────────────────────
def _clean_tag(name):
    return "".join(ch for ch in name if ch.isalnum())

def generate_report(p: dict) -> str:
    line = "━" * 6
    is_mixed = p["effect"] == MIXED_LABEL
    effect_name = "Mixed sequence" if is_mixed else p["effect"]
    rows = [
        f"Effect :: {effect_name}",
        f"Easing :: {p['easing']}",
        f"Format :: {p['format']}",
        f"Resolution :: {p['resolution']}",
        f"Duration :: {p['duration']} s",
        f"FPS :: {p['fps']}",
        f"Total frames :: {p['total_frames']}",
        f"Images :: {p['num_images']}",
        f"Transitions :: {p['num_transitions']}",
        f"Intensity :: {p['intensity']} / 50",
        f"Seed :: {p['seed']}",
        f"Encoding preset :: {p['ffmpeg_preset']}",
    ]
    if p.get("loop_mode") == "ping_pong":
        rows.append("Loop :: ping-pong")
    if p.get("mixed_effects"):
        rows.append("Mixed effects :: " + ", ".join(p["mixed_effects"]))

    if p.get("music_enabled") and p.get("bpm"):
        audio_rows = [
            f"Audio files :: {p['music_name']}",
            f"Avg BPM :: {p['bpm']:.1f}",
            f"Beats detected :: {p.get('n_beats_detected') or '—'}",
            f"Beat grid :: {p.get('beat_grid_label', '—')}",
            f"Track length :: {p['audio_duration']:.1f} s" if p.get("audio_duration") else "Track length :: —",
            f"Duration mode :: {'Full track' if p.get('full_song_mode') else 'Manual'}",
        ]
    else:
        audio_rows = ["No audio"]

    desc = ("A different random effect for each transition." if is_mixed
            else effect_descriptions.get(p["effect"], ""))
    tag = "MixedSequence" if is_mixed else _clean_tag(p["effect"])

    out = [f"FRAME-TO-FRAME :: {p.get('report_no', 1):03d}", line, "PARAMETERS", line]
    out += rows
    out += [line, "AUDIO", line]
    out += audio_rows
    out += ["", line, "DESCRIPTION", line, desc, "",
            f"#VideoArt #FrameToFrame #VisualFX #{tag} #TransitionFX"]
    return "\n".join(out)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
STATE_KEYS = ["video_bytes", "audio_bytes", "audio_name", "report_text", "video_ready",
              "preview_proxy", "preview_width"]
for key in STATE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "report_counter" not in st.session_state:
    st.session_state["report_counter"] = 0  # NON azzerato a ogni generazione

# ─────────────────────────────────────────────
# INTERFACCIA
# ─────────────────────────────────────────────
st.title("🎞️ Frame-to-Frame FX Video Generator 3.1 by Loop507")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        f"Carica da 2 a {MAX_IMAGES} immagini (max {MAX_UPLOAD_MB} MB in totale)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    n_up = len(uploaded_files) if uploaded_files else 0
    total_mb = sum(f.size for f in uploaded_files) / 1e6 if uploaded_files else 0.0
    upload_ok = 2 <= n_up <= MAX_IMAGES and total_mb <= MAX_UPLOAD_MB
    if n_up:
        st.caption(f"🧮 {n_up} / {MAX_IMAGES} immagini · {total_mb:.0f} / {MAX_UPLOAD_MB} MB")
        if n_up > MAX_IMAGES:
            st.error(f"Troppe immagini: il massimo è {MAX_IMAGES}. Rimuovine {n_up - MAX_IMAGES}.")
        elif total_mb > MAX_UPLOAD_MB:
            st.error(f"Peso totale {total_mb:.0f} MB sopra il limite di {MAX_UPLOAD_MB} MB: l'app rischia di "
                     f"andare in crash. Togli qualche foto o usa versioni più leggere.")

with col2:
    effect_choice = st.selectbox("Effetto", list(effect_funcs.keys()) + [MIXED_LABEL])
    easing_choice = st.selectbox("Curva easing", ["smootherstep", "smoothstep", "sinusoidal", "linear"])
    format_choice = st.selectbox("Formato video", ["1:1", "16:9", "9:16"])
    if effect_choice in AUDIO_EFFECTS or effect_choice == MIXED_LABEL:
        st.caption("🔊 Effetto audio-reattivo: segue kick, bassi, medi, alti e spettro del brano caricato. "
                   "Senza brano (e nell'anteprima) usa un ritmo sintetico a 120 BPM.")

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

if n_up >= 2 and not (use_music and audio_file is not None):
    min_s = (n_up - 1) * 4 / 24
    if duration < min_s:
        st.info(f"ℹ️ Con {n_up} immagini ogni transizione dura almeno 4 frame: il video durerà circa "
                f"{min_s:.1f} s invece di {duration} s. Per una durata più breve usa meno immagini.")

# ─────────────────────────────────────────────
# ANTEPRIMA RAPIDA
# ─────────────────────────────────────────────
prev_cols = st.columns([3, 2])
with prev_cols[0]:
    preview_btn = st.button("🔍 Anteprima rapida (bassa risoluzione)")
with prev_cols[1]:
    prev_n = st.radio("Immagini in anteprima", [4, 8], horizontal=True)

if preview_btn:
    if n_up < 2:
        st.warning("⚠️ Carica almeno 2 immagini per l'anteprima.")
    else:
        n_prev = min(prev_n, n_up)
        with st.spinner(f"Genero anteprima con {n_prev} immagini..."):
            if seed_input:
                random.seed(int(seed_input))
                np.random.seed(int(seed_input))
            prev_size = {"1:1": (240, 240), "16:9": (320, 180), "9:16": (180, 320)}[format_choice]
            prev_fps, prev_frames = 12, 18  # 1,5 s per transizione
            imgs_p = [load_image_fit(f, *prev_size) for f in uploaded_files[:n_prev]]
            counts_p = [prev_frames] * (n_prev - 1)
            path_p, written_p, mixed_p = build_video(
                imgs_p, list(range(n_prev)), counts_p, effect_choice, intensity, easing_choice,
                "ultrafast", prev_fps, random.Random(int(seed_input) or None), None, None, 5, None)
            with open(path_p, "rb") as f:
                data_p = f.read()
            _safe_remove(path_p)
            used = ", ".join(sorted(set(mixed_p))) if mixed_p else effect_choice
            note = "" if n_prev == prev_n else f" (ne hai caricate solo {n_up})"
            st.session_state["preview_quick"] = {
                "bytes": data_p, "width": prev_size[0],
                "caption": f"{n_prev} immagini{note} · {len(counts_p)} transizioni · {written_p / prev_fps:.1f} s · "
                           f"{prev_size[0]}×{prev_size[1]} · effetto: {used}"}

quick = st.session_state.get("preview_quick")
if quick:
    st.caption(quick["caption"])
    st.video(quick["bytes"], format="video/mp4", width=quick["width"])

# ─────────────────────────────────────────────
# GENERAZIONE
# ─────────────────────────────────────────────
video_btn = st.button("🎬 Genera Video")

if video_btn and upload_ok:

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

    images = [load_image_fit(f, *target_size) for f in uploaded_files]  # una per volta: RAM piatta
    image_sequence = build_image_sequence(len(images), loop_mode)
    num_transitions = len(images) - 1

    bpm_value = music_name = tmp_audio_path = audio_duration_sec = n_beats_detected = beat_times = audio_features = None
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

            bpm_value, beat_times, audio_duration_sec, n_beats_detected, audio_features = analyze_audio(tmp_audio_path, grid_mode, FPS)
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
        ffmpeg_preset, FPS, rnd, _on_progress, audio_features)
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
    status.text("Creo l'anteprima a bassa risoluzione...")
    st.session_state["preview_proxy"] = make_preview_proxy(final_video_path)
    st.session_state["preview_width"] = {"1:1": 400, "16:9": 520, "9:16": 280}[format_choice]
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
        "beat_grid_label": GRID_EN[grid_mode], "loop_mode": loop_mode, "ffmpeg_preset": ffmpeg_preset,
        "mixed_effects": sorted(set(mixed_effects_used)) if mixed_effects_used else None,
    }
    st.session_state["report_counter"] += 1
    report_params["report_no"] = st.session_state["report_counter"]
    st.session_state["report_text"] = generate_report(report_params)
    st.session_state["video_ready"] = True

    status.empty()
    progress.empty()
    st.success(f"✅ Video generato con successo! (seed {seed_used})")

elif video_btn:
    if n_up < 2:
        st.warning("⚠️ Carica almeno 2 immagini prima di generare.")
    else:
        st.warning(f"⚠️ Limiti da rispettare: massimo {MAX_IMAGES} immagini e {MAX_UPLOAD_MB} MB in totale "
                   f"(ora: {n_up} immagini, {total_mb:.0f} MB).")

# ─────────────────────────────────────────────
# DOWNLOAD — sempre visibili dopo la generazione (FUORI da st.button → non si resettano al rerun)
# ─────────────────────────────────────────────
if st.session_state.get("video_ready"):
    st.divider()
    st.subheader("▶️ Anteprima del video (bassa risoluzione)")
    if st.session_state.get("preview_proxy"):
        st.video(st.session_state["preview_proxy"], format="video/mp4", width=st.session_state["preview_width"])
        st.caption("Copia compressa solo per guardarlo qui, con l'audio: il file scaricato è in piena qualità.")
    else:
        st.caption("Anteprima non disponibile: scarica il video per vederlo.")
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
