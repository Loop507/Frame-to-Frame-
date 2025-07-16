import streamlit as st
import sys
import subprocess
from PIL import Image
import warnings

# Configurazione iniziale
st.set_page_config(page_title="Frame to Frame", layout="wide")

# Installazione fallback per OpenCV (solo se necessario)
try:
    import cv2
except ImportError:
    with st.spinner("Sto installando OpenCV..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "opencv-python-headless==4.9.0.80"
            ])
        import cv2

# Interfaccia principale
def main():
    st.title("🎥 Frame to Frame Processor")
    
    uploaded_file = st.file_uploader("Carica un video o immagine", 
                                   type=["jpg", "png", "mp4", "avi"])
    
    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            img = Image.open(uploaded_file)
            st.image(img, caption="Immagine caricata", use_column_width=True)
            
            # Esempio elaborazione con OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            st.image(gray, caption="Versione in grigio", clamp=True)
            
        else:
            st.warning("Elaborazione video non ancora implementata")

if __name__ == "__main__":
    main()
