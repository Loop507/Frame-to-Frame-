# 🎬 Video Effects Generator

**Frame to Frame by Loop507** - Crea video con effetti di transizione avanzati tra 2 o 3 foto

## ✨ Caratteristiche

- 🎨 **6 effetti di transizione**: Dissolvenza, Zoom, Pixel Random, Blocchi, Onda, Spirale
- 📸 **Supporto 2-3 foto**: Crea sequenze fluide tra multiple immagini
- ⏱️ **Durata personalizzabile**: Controllo preciso della durata del video
- 📊 **Barra di progresso**: Monitora lo stato di generazione in tempo reale
- 🎯 **Parametri flessibili**: Risoluzione, FPS, dimensioni blocchi personalizzabili
- 📝 **Testo sovrapposto**: Aggiungi testo personalizzato al video

## 🚀 Installazione

```bash
# Clona il repository
git clone https://github.com/tuousername/video-effects-generator
cd video-effects-generator

# Installa le dipendenze
pip install -r requirements.txt
```

## 📋 Dipendenze

- Python 3.7+
- numpy
- Pillow
- opencv-python
- tqdm

## 🎯 Utilizzo Base

### Video con 2 foto
```bash
python app.py --img1 foto1.jpg --img2 foto2.jpg --pixel-spiral
```

### Video con 3 foto
```bash
python app.py --img1 foto1.jpg --img2 foto2.jpg --img3 foto3.jpg --pixel-wave
```

## 🎨 Effetti Disponibili

### 1. 🌀 Spirale (`--pixel-spiral`)
Scambia pixel partendo dal centro con movimento spirale
```bash
python app.py --img1 photo1.jpg --img2 photo2.jpg --pixel-spiral --duration 8
```

### 2. 🌊 Onda (`--pixel-wave`)
Transizione fluida da sinistra a destra con effetto ondulatorio
```bash
python app.py --img1 photo1.jpg --img2 photo2.jpg --pixel-wave --duration 6
```

### 3. 🧩 Blocchi (`--pixel-blocks`)
Scambia blocchi di pixel di dimensioni personalizzabili
```bash
python app.py --img1 photo1.jpg --img2 photo2.jpg --pixel-blocks --block-size 16
```

### 4. 🎲 Pixel Casuali (`--pixel-random`)
Scambia pixel in modo casuale per un effetto granulare
```bash
python app.py --img1 photo1.jpg --img2 photo2.jpg --pixel-random --duration 10
```

### 5. 🎨 Dissolvenza (`--fade`)
Classico effetto di dissolvenza incrociata
```bash
python app.py --img1 photo1.jpg --img2 photo2.jpg --fade --duration 5
```

### 6. 🔍 Zoom (`--zoom`)
Effetto zoom progressivo durante la transizione
```bash
python app.py --img1 photo1.jpg --img2 photo2.jpg --zoom --zoom-factor 1.8
```

## ⚙️ Parametri Completi

```bash
python app.py [OPZIONI]
```

### Opzioni Obbligatorie
- `--img1 FILE` - Prima immagine (obbligatoria)
- `--img2 FILE` - Seconda immagine (obbligatoria)

### Opzioni Immagini
- `--img3 FILE` - Terza immagine (opzionale)

### Opzioni Video
- `--output FILE` - Nome file output (default: output.mp4)
- `--duration SECONDI` - Durata per transizione (sovrascrive --frames)
- `--frames NUM` - Frame per transizione (default: 150)
- `--fps NUM` - Frame per secondo (default: 30)
- `--width NUM` - Larghezza video (default: 640)
- `--height NUM` - Altezza video (default: 480)

### Opzioni Effetti
- `--fade` - Attiva dissolvenza
- `--zoom` - Attiva zoom progressivo
- `--pixel-random` - Scambio pixel casuali
- `--pixel-blocks` - Scambio a blocchi
- `--pixel-wave` - Effetto onda
- `--pixel-spiral` - Effetto spirale

### Opzioni Avanzate
- `--block-size NUM` - Dimensione blocchi (default: 8)
- `--zoom-factor FLOAT` - Fattore zoom (default: 1.5)
- `--text "TESTO"` - Aggiungi testo al video

## 💡 Esempi Pratici

### Video Instagram (1080x1080)
```bash
python app.py --img1 selfie1.jpg --img2 selfie2.jpg --pixel-spiral \
              --width 1080 --height 1080 --duration 6 --output instagram.mp4
```

### Video YouTube (1920x1080)
```bash
python app.py --img1 banner1.jpg --img2 banner2.jpg --pixel-wave \
              --width 1920 --height 1080 --duration 8 --output youtube.mp4
```

### Video con 3 foto e testo
```bash
python app.py --img1 foto1.jpg --img2 foto2.jpg --img3 foto3.jpg \
              --pixel-blocks --block-size 12 --duration 7 \
              --text "La mia storia" --output storia.mp4
```

### Video veloce per TikTok
```bash
python app.py --img1 pic1.jpg --img2 pic2.jpg --pixel-random \
              --width 1080 --height 1920 --duration 3 --fps 60 \
              --output tiktok.mp4
```

## 📊 Informazioni Durata

### Durate Default
- **2 foto**: 5 secondi totali (1 transizione × 5 sec)
- **3 foto**: 10 secondi totali (2 transizioni × 5 sec)

### Calcolo Personalizzato
```bash
# 8 secondi per transizione
--duration 8

# Controllo preciso con frame
--frames 240  # 8 secondi a 30fps
```

## 🎬 Formati Supportati

### Input
- **Immagini**: JPG, JPEG, PNG, BMP, TIFF
- **Dimensioni**: Qualsiasi (verranno ridimensionate)

### Output
- **Video**: MP4
- **Codec**: H.264 (mp4v)
- **Qualità**: Alta definizione

## 🛠️ Risoluzione Problemi

### Errore "File non trovato"
```bash
# Verifica che i file esistano
ls -la foto1.jpg foto2.jpg

# Usa percorsi assoluti se necessario
python app.py --img1 /path/completo/foto1.jpg --img2 /path/completo/foto2.jpg
```

### Errore OpenCV
```bash
# Su Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# Su macOS
brew install opencv
```

### Video non si apre
Assicurati di avere un player video aggiornato che supporti H.264

## 🔧 Sviluppo

### Aggiungere nuovi effetti
1. Crea una nuova funzione in `app.py`
2. Aggiungi il parametro argparse
3. Includi la logica nel main

### Contribuire
1. Fork del repository
2. Crea un branch per la feature
3. Commit delle modifiche
4. Push del branch
5. Crea una Pull Request

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT.

## 👨‍💻 Autore

**Loop507** - Frame to Frame Video Effects Generator

## 🌟 Esempi di Output

Il generatore produce video fluidi e professionali con transizioni cinematografiche. Perfetto per:

- 📱 Social media content
- 🎥 Presentazioni
- 🎨 Arte digitale
- 📊 Slideshow professionali
- 🎬 Video promozionali

---

⭐ **Lascia una stella se ti è piaciuto il progetto!**
