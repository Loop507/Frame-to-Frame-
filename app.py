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

# Qualità del video finale: (CRF x264, tetto di bitrate in Mbps). Misurato su 720p: il vecchio default
# (quality=8 di imageio = CRF 10) dava ~55 MB ogni 30 s con immagini lisce e fino a ~190 MB con immagini
# molto granulose. CRF + tetto tengono il file sotto controllo anche con effetti "rumorosi".
QUALITY_PRESETS = {
    "📦 Leggero (~9 MB ogni 30 s)": (26, 3.5),
    "⚖️ Bilanciato (~12 MB ogni 30 s)": (22, 6.0),
    "💎 Alta (~17 MB ogni 30 s)": (18, 12.0),
}

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

def _grid(h, w):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

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
def fade_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Dissolvenza a canali sfalsati: R segue i bassi, G i medi, B gli alti (ognuno anticipa o ritarda
    il proprio crossfade → il colore 'respira' con la musica). Ogni kick accende un bagliore (bloom)."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    spread = 0.25 + intensity / 60.0
    for k, a in enumerate(al):
        env, kick = _env(a), float(au.kick[k])
        e = np.array([au.bass[k], au.mid[k], au.high[k]], dtype=np.float32)
        ac = np.clip(a + spread * env * (e - 0.5) * 2.0, 0, 1).astype(np.float32)
        out = (f1 * (1 - ac) + f2 * ac) * (1.0 + 0.15 * kick * env)
        glow_amt = float(kick * env)
        if glow_amt > 0.1:
            g = cv2.GaussianBlur(cv2.resize(out, (max(1, w // 4), max(1, h // 4))), (0, 0), 5)
            out = out + cv2.resize(g, (w, h)) * (0.40 * glow_amt)
        yield np.clip(out, 0, 255).astype(np.uint8)

def morph_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Morph elastico a optical flow: il kick spinge (e a volte ritira) il movimento dei pixel come una
    molla, e gli acuti sfalsano la deformazione dei tre canali colore → scia cromatica lungo il flusso."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    f12, f21 = _flow_full(img1, img2), _flow_full(img2, img1)
    xs, ys = _grid(h, w)
    ch1, ch2 = cv2.split(img1), cv2.split(img2)
    for k, a in enumerate(al):
        env, kick, high = _env(a), float(au.kick[k]), float(au.high[k])
        push = 0.30 * (kick - 0.25) * env
        p1, p2 = a + push, (1 - a) - push
        scales = (1.0 + 0.20 * high * env, 1.0, 1.0 - 0.20 * high * env)
        w1 = cv2.merge([cv2.remap(c, xs - p1 * s * f12[..., 0], ys - p1 * s * f12[..., 1],
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c, s in zip(ch1, scales)])
        w2 = cv2.merge([cv2.remap(c, xs - p2 * s * f21[..., 0], ys - p2 * s * f21[..., 1],
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c, s in zip(ch2, scales)])
        yield _blend(w1, w2, a)

def glitch_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Glitch a strappi persistenti: ogni colpo fa nascere fasce orizzontali che restano 'incastrate'
    per qualche frame, scorrono di lato (ampiezza dai bassi), si dividono in RGB (acuti) e in parte
    mostrano l'ALTRA immagine. Tra un colpo e l'altro, micro-tremore di riga."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    amp = intensity / 15.0
    slices, prev = [], 0.0
    for k, a in enumerate(al):
        env = _env(a)
        hit, bass, high = float(au.onset[k]), float(au.bass[k]), float(au.high[k])
        remaining = len(al) - k - 2
        if hit > 0.4 and prev <= 0.4 and remaining > 0:
            for _ in range(int(round((2 + 10 * hit) * amp * env))):
                slices.append({"y": random.randint(0, max(0, h - 8)), "h": random.randint(4, max(5, int(h * 0.07))),
                               "dx": int(random.choice((-1, 1)) * random.uniform(0.02, 0.25) * w * (0.3 + bass)),
                               "ttl": min(random.randint(2, 7), remaining), "swap": random.random() < 0.4,
                               "split": int(2 + high * 20 * amp)})
            slices = slices[-30:]
        prev = hit
        frame = _blend(img1, img2, a)
        other = img2 if a < 0.5 else img1
        for s in slices:
            y0, y1 = s["y"], min(h, s["y"] + s["h"])
            src = np.roll((other if s["swap"] else frame)[y0:y1], s["dx"], axis=1)
            src[..., 0] = np.roll(src[..., 0], s["split"], axis=1)
            src[..., 2] = np.roll(src[..., 2], -s["split"], axis=1)
            frame[y0:y1] = src
            s["ttl"] -= 1
        slices = [s for s in slices if s["ttl"] > 0]
        reach = int(10 * amp) + 1
        for _ in range(int(env * (2 + 10 * high) * amp)):
            y = random.randint(0, h - 3)
            frame[y:y + 2] = np.roll(frame[y:y + 2], random.randint(-reach, reach), axis=1)
        yield frame

def pixelate_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Mosaico che respira: la dimensione dei pixel pulsa col kick e coi bassi, i colori vengono ridotti
    a una palette povera (più povera sui colpi) e sui colpi forti una parte dei blocchi si inverte."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    amp = min(1.0, intensity / 12.0 + 0.3)
    for k, a in enumerate(al):
        env, kick, bass, rms = _env(a), float(au.kick[k]), float(au.bass[k]), float(au.rms[k])
        frame = _blend(img1, img2, a)
        pulse = max(kick, 0.2 + 0.4 * rms)
        bs = int(1 + (4 + 56 * pulse * (0.5 + 0.5 * bass)) * env * amp)
        if bs > 1:
            small = cv2.resize(frame, (max(1, w // bs), max(1, h // bs)), interpolation=cv2.INTER_AREA)
            q = 256.0 / max(3, int(round(14 - 10 * pulse * env)))
            sm = np.floor(small.astype(np.float32) / q) * q + q / 2
            if kick > 0.55:
                inv = np.random.random(sm.shape[:2]) < (0.06 + 0.2 * (kick - 0.55))
                sm[inv] = 255 - sm[inv]
            frame = cv2.resize(np.clip(sm, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        yield frame

def corrupt_lines_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Databending: pixel dell'immagine vengono 'allungati' in strisce orizzontali e verticali (come un
    buffer corrotto). Il kick decide quante strisce, i bassi quanto sono lunghe; gli acuti aprono
    blocchi di rumore binario."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    amp = intensity / 15.0
    for k, a in enumerate(al):
        env, kick, bass, high = _env(a), float(au.kick[k]), float(au.bass[k]), float(au.high[k])
        frame = _blend(img1, img2, a)
        for _ in range(int((30 + 300 * kick) * env * amp)):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            col = (int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2]))
            length = int(random.uniform(0.02, 0.35) * w * (0.3 + bass))
            th = random.randint(2, 7)
            if random.random() < 0.75:
                cv2.line(frame, (x, y), (min(w - 1, x + length), y), col, th)
            else:
                cv2.line(frame, (x, y), (x, min(h - 1, y + int(length * h / w))), col, th)
        for _ in range(int(env * (1 + 8 * high) * amp)):
            bw, bh = random.randint(20, 200), random.randint(2, 12)
            x, y = random.randint(0, max(0, w - bw)), random.randint(0, max(0, h - bh))
            frame[y:y + bh, x:x + bw] = np.random.randint(0, 256, (min(bh, h - y), min(bw, w - x), 3), dtype=np.uint8)
        yield frame

def wave_distort_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Liquido: onde sinusoidali su entrambi gli assi. L'ampiezza segue bassi e kick, la lunghezza
    d'onda i medi, la velocità gli acuti; i tre canali colore ondeggiano con ampiezze diverse
    (aberrazione liquida)."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    xs, ys = _grid(h, w)
    y1d = np.arange(h, dtype=np.float32)[:, None]
    x1d = np.arange(w, dtype=np.float32)[None, :]
    p1 = p2 = 0.0
    for k, a in enumerate(al):
        env, kick, bass, mid, high = _env(a), float(au.kick[k]), float(au.bass[k]), float(au.mid[k]), float(au.high[k])
        p1 += 0.10 + 0.5 * high
        p2 += 0.07 + 0.4 * mid
        A = intensity * (0.6 + 2.2 * bass + 1.5 * kick) * env
        dx = (A * np.sin(2 * np.pi * y1d / (h / (2 + 6 * mid)) + p1)).astype(np.float32)
        dy = (0.6 * A * np.sin(2 * np.pi * x1d / (w / (2 + 5 * mid)) + p2)).astype(np.float32)
        chans = [cv2.remap(c, xs - dx * f, ys - dy * f, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                 for c, f in zip(cv2.split(_blend(img1, img2, a)), (1.15, 1.0, 0.85))]
        yield cv2.merge(chans)

def zoom_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Zoom blur radiale: ad ogni kick l'inquadratura 'spara' verso il centro con strisce di velocità
    (7 copie a zoom crescente fuse insieme), con una leggera rotazione guidata dai medi.
    Rientra a 1× a fine transizione."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    center = (w / 2.0, h / 2.0)
    taps = 7
    for k, a in enumerate(al):
        env, kick, mid = _env(a), float(au.kick[k]), float(au.mid[k])
        frame = _blend(img1, img2, a)
        z = (intensity / 50.0) * env * (0.25 + 1.2 * kick)
        ang = (mid - 0.5) * env * (intensity / 10.0)
        acc = np.zeros(frame.shape, np.float32)
        last = frame
        for t in range(taps):
            f = t / (taps - 1)
            M = cv2.getRotationMatrix2D(center, ang * f, 1.0 + z * f)
            last = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            acc += last
        yield cv2.addWeighted((acc / taps).astype(np.uint8), 0.7, last, 0.3, 0)

def zoom_random_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    h, w = img1.shape[:2]
    for a in _alphas(num_frames, easing, alphas):
        zoom = 1.0 + random.uniform(0, intensity / 50.0) * _env(a)
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        z1 = cv2.warpAffine(img1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        z2 = cv2.warpAffine(img2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        yield _blend(z1, z2, a)

def wipe_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Wipe a spettro: il fronte avanza da sinistra a destra, ma ogni riga lo sposta in base alla SUA
    banda di frequenza (il bordo diventa un'onda che balla), il volume allarga la sfumatura e il
    kick accende una linea luminosa lungo il fronte."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    xs = np.arange(w, dtype=np.float32)[None, :]
    ridx = (np.arange(h) * N_BANDS) // h
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    for k, a in enumerate(al):
        env, kick, rms = _env(a), float(au.kick[k]), float(au.rms[k])
        feather = max(4.0, intensity * (0.4 + rms))
        e = au.bands[k][ridx].astype(np.float32)
        edge = (-feather + a * (w + 2 * feather) + intensity * 6.0 * env * (e - 0.4) * (0.5 + rms)).astype(np.float32)[:, None]
        m = np.clip((edge - xs) / feather + 1, 0, 1)[:, :, None]
        glow = np.exp(-((xs - edge) / (feather * 0.7)) ** 2)[:, :, None] * ((0.35 + 0.65 * kick) * env * 160.0)
        yield np.clip(f1 * (1 - m) + f2 * m + glow, 0, 255).astype(np.uint8)

def diagonal_wipe_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Persiane diagonali: 14 strisce che spazzano in versi alterni; ogni striscia ritarda in base
    all'energia della propria banda di frequenza e il kick accende i bordi."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    n, soft, stag = 14, 0.10, 0.35
    xx, yy = _grid(h, w)
    s_idx = np.clip(((xx - yy + h) / (w + h) * n).astype(np.int32), 0, n - 1)
    diag = (xx + yy) / (w + h)
    dcoord = np.where(s_idx % 2 == 1, 1.0 - diag, diag).astype(np.float32)
    band_pos = np.linspace(0, N_BANDS - 1, n)
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    for k, a in enumerate(al):
        env, kick = _env(a), float(au.kick[k])
        e = np.interp(band_pos, np.arange(N_BANDS), au.bands[k]).astype(np.float32)
        o = stag * env * e
        p = np.clip((a - o) / (1.0 - o), 0, 1).astype(np.float32)
        m = np.clip((p[s_idx] * (1 + soft) - dcoord) / soft, 0, 1)
        edge = (4.0 * m * (1.0 - m) * ((0.3 + 0.7 * kick) * env * 200.0))[:, :, None]
        m3 = m[:, :, None]
        yield np.clip(f1 * (1 - m3) + f2 * m3 + edge, 0, 255).astype(np.uint8)

def iris_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Iris a spettro: il bordo del cerchio ha un raggio diverso per ogni angolo (spettro specchiato
    attorno al centro) → una stella/fiore che pulsa; il kick accende l'anello."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    cx, cy = w / 2.0, h / 2.0
    xs, ys = _grid(h, w)
    dx, dy = xs - cx, ys - cy
    dist = np.sqrt(dx * dx + dy * dy)
    u = np.abs((np.arctan2(dy, dx) + np.pi) / np.pi - 1.0)
    idx = np.minimum((u * N_BANDS).astype(np.int32), N_BANDS - 1)
    max_r = float(np.sqrt(cx ** 2 + cy ** 2))
    feather = max(4.0, float(intensity))
    amp = min(0.7, 0.2 + intensity / 50.0)
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    for k, a in enumerate(al):
        env, kick = _env(a), float(au.kick[k])
        shape = (1.0 + amp * env * (au.bands[k].astype(np.float32) - 0.45) * 2.0).astype(np.float32)
        R = (-feather + a * (max_r * 1.35 + feather)) * shape[idx]
        m = np.clip((R - dist) / feather + 1, 0, 1)[:, :, None]
        ring = np.exp(-((dist - R) / (feather * 0.8)) ** 2)[:, :, None] * ((0.3 + 0.7 * kick) * env * 180.0)
        yield np.clip(f1 * (1 - m) + f2 * m + ring, 0, 255).astype(np.uint8)

def color_echo_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Eco cromatico nel TEMPO: R, G e B pescano da frame passati diversi (R ritarda coi bassi, G coi
    medi, B con gli acuti) e scivolano di lato in verso opposto sui colpi. Un vero eco, non un alone."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    hist = []
    for k, a in enumerate(al):
        env, kick = _env(a), float(au.kick[k])
        hist = (hist + [_blend(img1, img2, a)])[-18:]
        maxd = min(len(hist) - 1, int(3 + intensity / 4))
        dR = int(round(float(au.bass[k]) * maxd * env))
        dG = int(round(float(au.mid[k]) * maxd * env * 0.5))
        dB = int(round(float(au.high[k]) * maxd * env))
        sh = int(intensity * 0.4 * env * (0.3 + kick))
        r, g, b = hist[-1 - dR][..., 0], hist[-1 - dG][..., 1], hist[-1 - dB][..., 2]
        shr, shb = sh + dR * 4, sh + dB * 4
        if shr:
            r = np.roll(r, shr, axis=1)
        if shb:
            b = np.roll(b, -shb, axis=1)
        yield cv2.merge([np.ascontiguousarray(r), np.ascontiguousarray(g), np.ascontiguousarray(b)])

def particle_float_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Scintille: ogni kick fa esplodere uno sciame di particelle con scia, colorate dai pixel
    dell'immagine sotto (schiariti). Il volume dà la velocità, i bassi lo spessore, i medi la
    gravità; un alone sfocato le fa brillare."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    max_n, prev = 700, 0.0
    pos, vel = np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    ttl, col = np.zeros(0, np.int32), np.zeros((0, 3), np.uint8)
    for k, a in enumerate(al):
        env = _env(a)
        kick, rms, bass, mid = float(au.kick[k]), float(au.rms[k]), float(au.bass[k]), float(au.mid[k])
        frame = _blend(img1, img2, a)
        remaining = len(al) - k - 1
        if remaining > 2 and ((kick > 0.5 and prev <= 0.5) or random.random() < 0.15 * rms * env):
            nb = max(4, int((40 + 240 * kick) * min(1.5, intensity / 15.0) * max(env, 0.25)))
            c = np.array([random.uniform(0.1 * w, 0.9 * w), random.uniform(0.1 * h, 0.9 * h)], np.float32)
            p = c + np.random.normal(0, 18, (nb, 2)).astype(np.float32)
            ang = np.random.uniform(0, 2 * np.pi, nb).astype(np.float32)
            spd = (np.random.uniform(3, 16, nb) * (0.5 + rms)).astype(np.float32)
            v = np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None]
            px, py = np.clip(p[:, 0], 0, w - 1).astype(int), np.clip(p[:, 1], 0, h - 1).astype(int)
            cl = np.clip(frame[py, px].astype(np.float32) * 1.5 + 60, 0, 255).astype(np.uint8)
            t = np.minimum(np.random.randint(8, 22, nb), remaining).astype(np.int32)
            pos, vel = np.concatenate([pos, p])[-max_n:], np.concatenate([vel, v])[-max_n:]
            ttl, col = np.concatenate([ttl, t])[-max_n:], np.concatenate([col, cl])[-max_n:]
        prev = kick
        layer = np.zeros_like(frame)
        if len(pos):
            new = pos + vel
            th = 2 + int(3 * bass)
            for (x0, y0), (x1, y1), c in zip(pos, new, col):
                cv2.line(layer, (int(x0), int(y0)), (int(x1), int(y1)), (int(c[0]), int(c[1]), int(c[2])), th)
            pos, vel, ttl = new, vel * 0.96, ttl - 1
            vel[:, 1] += (mid - 0.5) * 0.5
            keep = (ttl > 0) & (pos[:, 0] >= 0) & (pos[:, 0] < w) & (pos[:, 1] >= 0) & (pos[:, 1] < h)
            pos, vel, ttl, col = pos[keep], vel[keep], ttl[keep], col[keep]
        glow = cv2.resize(cv2.GaussianBlur(cv2.resize(layer, (max(1, w // 4), max(1, h // 4))), (0, 0), 3), (w, h))
        yield cv2.add(cv2.add(frame, layer), cv2.addWeighted(glow, 1.6, glow, 0, 0))

def film_look_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Proiettore impazzito: il fotogramma ballonzola nel gate coi bassi, l'esposizione lampeggia sui
    kick (con bruciatura arancio ai colpi forti), la grana cresce con gli acuti, la vignettatura
    con il volume e le righe verticali di graffi compaiono sugli attacchi."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    xs, ys = _grid(h, w)
    r2 = (((xs - w / 2.0) / (w / 2.0)) ** 2 + ((ys - h / 2.0) / (h / 2.0)) ** 2) / 2.0
    prev = 0.0
    for k, a in enumerate(al):
        env = _env(a)
        kick, bass, high, rms, hit = (float(au.kick[k]), float(au.bass[k]), float(au.high[k]),
                                      float(au.rms[k]), float(au.onset[k]))
        frame = _blend(img1, img2, a)
        wy, wx = random.uniform(-1, 1) * bass * env * intensity * 0.5, random.uniform(-1, 1) * bass * env * intensity * 0.3
        frame = cv2.warpAffine(frame, np.float32([[1, 0, wx], [0, 1, wy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
        out = frame.astype(np.float32) * (1.0 + (random.uniform(-0.05, 0.05) + 0.25 * kick) * env)
        out *= (1.0 - (0.25 + 0.5 * rms) * env * r2)[:, :, None]
        burn = float(np.clip((kick - 0.55) * 3.0, 0, 1) * env)
        if burn > 0.05:
            bx = random.choice((0.0, float(w)))
            blob = np.exp(-(((xs - bx) / (w * 0.35)) ** 2 + ((ys - h * random.uniform(0.2, 0.8)) / (h * 0.5)) ** 2))
            out += blob[:, :, None] * np.array([255, 150, 60], np.float32) * (0.7 * burn)
        sigma = intensity * (0.3 + high) * env
        if sigma > 0.5:
            grain = cv2.resize(np.random.standard_normal((max(1, h // 2), max(1, w // 2))).astype(np.float32) * sigma,
                               (w, h), interpolation=cv2.INTER_LINEAR)
            out += grain[:, :, None]
        out = np.clip(out, 0, 255).astype(np.uint8)
        if hit > 0.4 and prev <= 0.4:
            for _ in range(int(1 + 5 * hit * env)):
                x, y0 = random.randint(0, w - 1), random.randint(0, h // 2)
                cv2.line(out, (x, y0), (x + random.randint(-2, 2), y0 + random.randint(h // 4, h // 2)), (225, 225, 225), 1)
        prev = hit
        yield out

def vhs_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Nastro rovinato: il kick fa 'slittare' il quadro in verticale (molla), i bassi allargano la banda di
    tracking che strappa le righe, il volume sporca il colore (bleeding R/B), gli acuti aggiungono
    righe di rumore e il rumore da testina in fondo al quadro."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    xs, ys = _grid(h, w)
    roll_pos = roll_vel = phase = prev = 0.0
    amp = intensity / 15.0
    for k, a in enumerate(al):
        env = _env(a)
        kick, bass, high, rms = float(au.kick[k]), float(au.bass[k]), float(au.high[k]), float(au.rms[k])
        if kick > 0.55 and prev <= 0.55:
            roll_vel += random.choice((-1, 1)) * h * 0.04 * kick
        prev = kick
        roll_pos = (roll_pos + roll_vel) * 0.8
        roll_vel *= 0.7
        frame = np.roll(_blend(img1, img2, a), int(roll_pos * env), axis=0)
        jit = cv2.GaussianBlur(np.random.standard_normal((h, 1)).astype(np.float32), (1, 0), 6) * (1.0 + 6.0 * high) * env * 3.0
        phase += 0.15
        band_c = int(h * (0.80 + 0.15 * np.sin(phase)))
        bh = int(h * 0.04 * (1 + 3 * bass) * env)
        if bh > 0:
            lo = max(0, band_c - bh // 2)
            jit[lo:lo + bh, 0] += np.random.uniform(-1, 1, len(jit[lo:lo + bh, 0])) * (20 + 80 * rms) * amp
        frame = cv2.remap(frame, xs + jit, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        c = int((2 + 8 * rms) * env * amp)
        r, g, b = cv2.split(frame)
        if c:
            r = cv2.blur(np.roll(r, c, axis=1), (9, 1))
            b = cv2.blur(np.roll(b, -c, axis=1), (9, 1))
        frame = cv2.merge([r, g, b])
        nh = max(1, int(h * 0.03 * env))
        if env > 0.05:
            frame[h - nh:] = cv2.addWeighted(frame[h - nh:], 0.3, np.random.randint(0, 256, (nh, w, 3), dtype=np.uint8), 0.7, 0)
        for _ in range(int(env * (2 + 20 * high))):
            y, x = random.randint(0, h - 1), random.randint(0, w - 40)
            cv2.line(frame, (x, y), (x + random.randint(10, 40), y), (235, 235, 235), 1)
        yield frame

def motion_blur_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Strisciata direzionale: la lunghezza della scia esplode sui kick, la direzione ruota coi medi,
    e le alte luci vengono tirate in strisce anamorfiche (alone orizzontale azzurrino) sui colpi."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    hw, hh = max(1, w // 2), max(1, h // 2)
    qw, qh = max(1, w // 4), max(1, h // 4)
    kq = max(3, int(qw * 0.4)) | 1
    angle, taps = 0.0, 9
    for k, a in enumerate(al):
        env, kick, mid, rms = _env(a), float(au.kick[k]), float(au.mid[k]), float(au.rms[k])
        frame = _blend(img1, img2, a)
        angle += 0.05 + 0.3 * mid
        L = (0.01 + 0.10 * kick + 0.04 * rms) * w * env * (intensity / 15.0)
        small = cv2.resize(frame, (hw, hh), interpolation=cv2.INTER_AREA)
        acc = np.zeros(small.shape, np.float32)
        for t in np.linspace(-0.5, 0.5, taps):
            M = np.float32([[1, 0, t * L * np.cos(angle) / 2.0], [0, 1, t * L * np.sin(angle) / 2.0]])
            acc += cv2.warpAffine(small, M, (hw, hh), borderMode=cv2.BORDER_REPLICATE)
        blurred = cv2.resize((acc / taps).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        m = float(np.clip(env * (0.4 + 0.6 * kick), 0, 1))
        out = cv2.addWeighted(frame, 1 - m, blurred, m, 0)
        streak = float(kick * env)
        if streak > 0.1:
            q = cv2.resize(frame, (qw, qh), interpolation=cv2.INTER_AREA).astype(np.float32)
            hi = q * (q.mean(axis=2, keepdims=True) > 175)
            hi = cv2.blur(hi, (kq, 1)) * np.array([0.7, 0.95, 1.25], np.float32)
            out = cv2.add(out, cv2.resize(np.clip(hi * (0.8 * streak), 0, 255).astype(np.uint8), (w, h)))
        yield out

def scanlines_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Monitor CRT: lo schermo si CURVA (barrel) sui kick e sui bassi, le scanline si scavano col volume
    e scorrono in verticale sui colpi, la maschera RGB a triadi si accende e le alte luci
    sbavano (bloom)."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    cx, cy = w / 2.0, h / 2.0
    xs, ys = _grid(h, w)
    ddx, ddy = xs - cx, ys - cy
    r2 = (ddx / cx) ** 2 + (ddy / cy) ** 2
    y1d = np.arange(h, dtype=np.float32)
    tri = np.array([[1, .55, .55], [.55, 1, .55], [.55, .55, 1]], np.float32)[np.arange(w) % 3]
    roll, prev = 0.0, 0.0
    amp = min(1.0, intensity / 20.0)
    for k, a in enumerate(al):
        env, kick, bass, rms = _env(a), float(au.kick[k]), float(au.bass[k]), float(au.rms[k])
        if kick > 0.55 and prev <= 0.55:
            roll += random.uniform(1, 3)
        prev = kick
        kb = (0.04 + 0.28 * kick + 0.10 * bass) * (env ** 1.5) * amp * 1.5
        frame = _blend(img1, img2, a)
        warped = cv2.remap(frame, (cx + ddx * (1 + kb * r2)).astype(np.float32), (cy + ddy * (1 + kb * r2)).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        out = warped.astype(np.float32)
        d = (0.3 + 0.5 * rms) * env
        out *= (1 - d * 0.5 * (1 + np.cos(2 * np.pi * (y1d + roll) / 3.0)))[:, None, None]
        out *= (1 - 0.35 * env * amp + 0.35 * env * amp * tri)[None, :, :]
        bloom = cv2.resize(cv2.GaussianBlur(cv2.resize(warped, (max(1, w // 4), max(1, h // 4))), (0, 0), 4), (w, h))
        out += bloom.astype(np.float32) * ((0.2 + 0.5 * kick) * env)
        out *= 1.0 + 0.04 * np.sin(k * 1.7) * env
        yield np.clip(out, 0, 255).astype(np.uint8)

def burn_in_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Burn-in: la prima immagine resta 'bruciata' sullo schermo come fosforo, si allarga e scivola
    (i bassi la trascinano) mentre svanisce; ogni kick imprime un nuovo fantasma ambra del frame
    corrente. Fusione 'screen', così il doppione si vede sopra l'immagine nuova."""
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    tint = np.array([1.0, 0.78, 0.45], np.float32)
    center = (w / 2.0, h / 2.0)
    P, prev = None, 0.0
    for k, a in enumerate(al):
        env, kick, bass, rms = _env(a), float(au.kick[k]), float(au.bass[k]), float(au.rms[k])
        cur = _blend(img1, img2, a).astype(np.float32)
        if P is None or (kick > 0.55 and prev <= 0.55):
            P = cur.copy()
        prev = kick
        M = cv2.getRotationMatrix2D(center, 0, 1.0 + 0.012 + 0.03 * rms)
        M[0, 2] += 4 + bass * 14
        P = cv2.warpAffine(P, M, (w, h), borderMode=cv2.BORDER_REPLICATE) * (0.88 + 0.08 * rms)
        g = float(np.clip(env * min(1.0, intensity / 15.0 + 0.4), 0, 1))
        ghost = np.clip(P * tint * 1.5, 0, 255)
        screened = 255.0 - (255.0 - cur) * (255.0 - ghost) / 255.0
        yield np.clip(cur * (1 - g) + screened * g, 0, 255).astype(np.uint8)

def light_leaks_effect(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    """Perdite di luce spettrali: tre aloni morbidi si muovono in orbite lente, il loro colore deriva con
    gli acuti e la luminosità segue il volume; ogni kick spara un flare bianco-giallo con striscia
    orizzontale. Fusione 'screen' come su pellicola."""
    import colorsys
    h, w = img1.shape[:2]
    al = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(al))
    sw, sh = max(1, w // 4), max(1, h // 4)
    gx, gy = np.meshgrid(np.linspace(0, 1, sw, dtype=np.float32), np.linspace(0, 1, sh, dtype=np.float32))
    leaks = [{"ph": random.uniform(0, 6.28), "sp": random.uniform(0.05, 0.15), "hue": random.uniform(0, 0.12)} for _ in range(3)]
    flare, prev = 0.0, 0.0
    fx, fy = 0.5, 0.5
    for k, a in enumerate(al):
        env, kick, rms, bass, high = _env(a), float(au.kick[k]), float(au.rms[k]), float(au.bass[k]), float(au.high[k])
        if kick > 0.55 and prev <= 0.55:
            flare, fx, fy = 1.0, random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        prev = kick
        frame = _blend(img1, img2, a)
        layer = np.zeros((sh, sw, 3), np.float32)
        for i, lk in enumerate(leaks):
            cx = 0.5 + 0.55 * np.sin(lk["ph"] + k * lk["sp"] * (1 + i))
            cy = 0.5 + 0.55 * np.cos(lk["ph"] * 1.3 + k * lk["sp"])
            sig = 0.25 * (0.7 + bass)
            g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sig * sig))[:, :, None]
            col = np.array(colorsys.hsv_to_rgb((lk["hue"] + 0.3 * high + 0.2 * i) % 1.0, 0.85, 1.0), np.float32)
            layer += g * col * (255.0 * (0.12 + 0.35 * rms + 0.25 * kick) * env)
        if flare > 0.05:
            fl = np.exp(-(((gx - fx) / 0.35) ** 2 + ((gy - fy) / 0.035) ** 2) / 2.0) + \
                 np.exp(-(((gx - fx) ** 2 + (gy - fy) ** 2) / (2 * 0.06 ** 2)))
            layer += fl[:, :, None] * np.array([255, 235, 170], np.float32) * (0.7 * flare * env)
            flare *= 0.7
        leak = cv2.resize(np.clip(layer, 0, 255), (w, h), interpolation=cv2.INTER_LINEAR)
        yield np.clip(255.0 - (255.0 - frame.astype(np.float32)) * (255.0 - leak) / 255.0, 0, 255).astype(np.uint8)


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

AUDIO_EFFECTS = set()  # riempito dopo effect_funcs

# ── Combo: UNA curva alpha globale tagliata in fette → continuità + frame esatti ──
def _combo(funcs, img1, img2, num_frames, intensity, easing, alphas, audio=None):
    full = _alphas(num_frames, easing, alphas)
    au = _au(audio, len(full))
    start = 0
    for fn, part in zip(funcs, np.array_split(np.asarray(full), len(funcs))):
        if len(part) == 0:
            continue
        yield from fn(img1, img2, len(part), intensity, easing, alphas=part, audio=au.slice(start, len(part)))
        start += len(part)

def combo_effect_preset1(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    yield from _combo((glitch_effect, pixelate_effect), img1, img2, num_frames, intensity, easing, alphas, audio)

def combo_effect_preset2(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    yield from _combo((wave_distort_effect, color_echo_effect, particle_float_effect),
                      img1, img2, num_frames, intensity, easing, alphas, audio)

def combo_effect_preset3(img1, img2, num_frames, intensity, easing, alphas=None, audio=None):
    yield from _combo((motion_blur_effect, film_look_effect, zoom_random_effect),
                      img1, img2, num_frames, intensity, easing, alphas, audio)

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

AUDIO_EFFECTS.update(k for k in effect_funcs if k != "Zoom Random")  # tutti reattivi tranne Zoom Random

effect_descriptions = {
    "Fade": "Audio-reactive dissolve: R, G and B crossfade at different times (bass, mids, highs) and every kick lights a bloom.",
    "Morph": "Audio-reactive elastic optical-flow morph: kicks push and pull the pixel motion, highs split the three color channels along the flow.",
    "Glitch": "Audio-reactive glitch: each hit spawns stuck horizontal slices that slide sideways, split in RGB and sometimes show the other image.",
    "Pixelate": "Audio-reactive mosaic: pixel size pulses with kick and bass, the palette collapses on hits and blocks invert on strong ones.",
    "Corrupt Lines": "Audio-reactive databending: pixels are stretched into horizontal and vertical streaks, bass sets their length, highs open noise blocks.",
    "Wave Distort": "Audio-reactive liquid waves on both axes: bass and kick set the amplitude, mids the wavelength, highs the speed, with per-channel aberration.",
    "Zoom": "Audio-reactive radial zoom blur: every kick shoots the frame toward the center with speed streaks and a slight mid-driven rotation.",
    "Zoom Random": "Zoom from a random focus for a chaotic effect, retracts at the end of the transition.",
    "Wipe Direzionale": "Audio-reactive spectrum wipe: each row moves the front by its own frequency band, a glowing line rides the edge on kicks.",
    "Wipe Diagonale": "Audio-reactive diagonal blinds: 14 stripes sweep in alternating directions, each lagging by its band energy, edges flash on kicks.",
    "Iris Circolare": "Audio-reactive spectrum iris: the circle edge takes a different radius at every angle, a pulsing star shaped by the spectrum.",
    "Color Echo": "Audio-reactive temporal echo: R, G and B pull from different past frames (bass, mids, highs) and slide apart on hits.",
    "Particle Float": "Audio-reactive sparks: every kick bursts a swarm of trailing particles colored by the image, with a glow.",
    "Film Look": "Audio-reactive broken projector: gate weave with bass, exposure flashes and orange burns on kicks, grain with highs, scratches on attacks.",
    "VHS": "Audio-reactive damaged tape: kicks slip the picture vertically, bass widens the tracking tear, volume bleeds the color, highs add noise lines.",
    "Motion Blur": "Audio-reactive directional smear: streak length explodes on kicks, direction rotates with mids, highlights stretch into anamorphic streaks.",
    "Scanlines": "Audio-reactive CRT: the screen curves on kicks and bass, scanlines deepen with volume, RGB triad mask and phosphor bloom.",
    "Burn-in": "Audio-reactive burn-in: the first image stays burned like phosphor and fades slowly, every kick stamps a new amber ghost that bass drags sideways.",
    "Light Leaks": "Audio-reactive light leaks: three orbiting glows that drift in color with highs and brighten with volume, kicks fire a streaked flare.",
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
    "Combo Creative 1": "Audio-reactive Glitch + Pixelate in sequence (continuous alpha curve).",
    "Combo Creative 2": "Audio-reactive Wave Distort + Color Echo + Particle Float in sequence.",
    "Combo Creative 3": "Audio-reactive Motion Blur + Film Look + Zoom Random in sequence.",
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
def load_audio_streaming(path, target_sr=22050, block_seconds=30, max_seconds=None):
    """Decodifica a blocchi (mono, 22.05 kHz): la RAM è ~0.09 MB per secondo di brano, mentre
    librosa.load tiene l'intero file decodificato stereo a piena frequenza (~0.7 MB/s in transitorio)."""
    try:
        import soundfile as sf
        import soxr
        with sf.SoundFile(path) as f:
            sr_in = f.samplerate
            rs = soxr.ResampleStream(sr_in, target_sr, 1, dtype="float32") if sr_in != target_sr else None
            out, limit = [], (int(max_seconds * target_sr) if max_seconds else None)
            for block in f.blocks(blocksize=int(block_seconds * sr_in), dtype="float32", always_2d=True):
                mono = np.ascontiguousarray(block.mean(axis=1), dtype=np.float32)
                out.append(rs.resample_chunk(mono) if rs else mono)
                if limit and sum(len(o) for o in out) >= limit:
                    break
            if rs:
                out.append(rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
        y = np.concatenate(out)
        return (y[:limit] if limit else y), target_sr
    except Exception:
        import librosa  # formati/versioni non gestiti da soundfile: percorso classico
        return librosa.load(path, sr=target_sr, mono=True, duration=max_seconds)

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

def analyze_audio(audio_path, grid_mode, fps=24, max_seconds=None):
    import librosa
    hop = 512
    y, sr = load_audio_streaming(audio_path, 22050, max_seconds=max_seconds)
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
                ffmpeg_preset, fps, rnd, on_progress=None, features=None, crf=22, macro_block_size=16, maxrate=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
    # CRF esplicito (quality=None: imageio non ne aggiunge uno suo; con quality=8 sarebbe CRF 10 → file enormi)
    # + tetto di bitrate opzionale (CRF vincolato): il peso massimo del file è noto in anticipo
    params = ["-preset", ffmpeg_preset, "-crf", str(int(crf))]
    if maxrate:
        params += ["-maxrate", f"{maxrate}M", "-bufsize", f"{2 * maxrate}M"]
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=None,
                                macro_block_size=macro_block_size, output_params=params)
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
            kwargs = {"audio": features.slice(cursor, n_fr)}
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
    quality_label = st.radio("Qualità / peso del file", list(QUALITY_PRESETS), index=1, horizontal=True,
                             help="Stime misurate a 720p con immagini lisce. Con foto molto granulose il peso sale fino a "
                                  "circa 13 / 23 / 44 MB ogni 30 s (il tetto di bitrate lo limita).")
    crf_sel, maxrate_sel = QUALITY_PRESETS[quality_label]

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
            # con un brano caricato: feature audio vere dei primi 16 s (in cache) + audio nell'anteprima
            features_p, audio_path_p = None, None
            if use_music and audio_file is not None:
                try:
                    raw_a = audio_file.getvalue()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as ta:
                        ta.write(raw_a)
                        audio_path_p = ta.name
                    ckey = (audio_file.name, len(raw_a))
                    cache = st.session_state.get("preview_audio_cache")
                    if cache and cache["key"] == ckey:
                        features_p = cache["features"]
                    else:
                        features_p = analyze_audio(audio_path_p, "normal", prev_fps, max_seconds=16.0)[4]
                        st.session_state["preview_audio_cache"] = {"key": ckey, "features": features_p}
                except Exception as e:
                    st.warning(f"⚠️ Non riesco ad analizzare il brano per l'anteprima ({e}): uso il ritmo sintetico.")
                    features_p = None
            path_p, written_p, mixed_p = build_video(
                imgs_p, list(range(n_prev)), counts_p, effect_choice, intensity, easing_choice,
                "ultrafast", prev_fps, random.Random(int(seed_input) or None), None, features_p, 30, None)
            final_p = path_p
            if audio_path_p and features_p is not None:
                muxed_p, _err = mux_audio(path_p, audio_path_p)
                final_p = muxed_p or path_p
            with open(final_p, "rb") as f:
                data_p = f.read()
            _safe_remove(path_p, final_p if final_p != path_p else None, audio_path_p)
            used = ", ".join(sorted(set(mixed_p))) if mixed_p else effect_choice
            note = "" if n_prev == prev_n else f" (ne hai caricate solo {n_up})"
            audio_note = " · audio: il tuo brano" if features_p is not None else " · ritmo sintetico"
            st.session_state["preview_quick"] = {
                "bytes": data_p, "width": prev_size[0],
                "caption": f"{n_prev} immagini{note} · {len(counts_p)} transizioni · {written_p / prev_fps:.1f} s · "
                           f"{prev_size[0]}×{prev_size[1]} · effetto: {used}{audio_note}"}

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
        ffmpeg_preset, FPS, rnd, _on_progress, audio_features, crf=crf_sel, maxrate=maxrate_sel)
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
            label=f"📥 Scarica Video (.mp4) · {len(st.session_state['video_bytes']) / 1e6:.1f} MB",
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
