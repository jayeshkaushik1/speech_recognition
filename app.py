import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import os
# FORCE LEGACY KERAS (Fixes "bad marshal data" or "unknown layer" errors)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
# ... rest of your code
# Debugging: Print all files in the current directory
st.write("Current Directory:", os.getcwd())
st.write("Files found:", os.listdir())
# --- Config ---
FRAME_LENGTH = 2048
HOP_LENGTH = 512
TARGET_LENGTH = 180000
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")

# --- Cache the model loading so it doesn't reload on every click ---
@st.cache_resource
def load_models():
    try:
        # Check if files exist to prevent crashing
        if not os.path.exists('model_female.h5') or not os.path.exists('model_male.h5'):
            return None, None
        
        model_f = tf.keras.models.load_model('model_female.h5')
        model_m = tf.keras.models.load_model('model_male.h5')
        return model_f, model_m
    except Exception as e:
        return None, None

def process_audio(file_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. Trim Silence
    trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # 3. Pad/Truncate to 180000
    if len(trimmed) > TARGET_LENGTH:
        padded = trimmed[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(trimmed)
        padded = np.pad(trimmed, (0, padding), 'constant')
    
    # 4. Extract Features
    zcr = librosa.feature.zero_crossing_rate(padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=padded, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    
    # 5. Stack Features (1, 352, 15)
    features = np.concatenate((zcr.T, rms.T, mfccs.T), axis=1)
    features = np.expand_dims(features, axis=0)
    
    return features.astype('float32')

# --- UI Layout ---
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV file to detect the emotion.")

# Load Models
model_female, model_male = load_models()

if model_female is None:
    st.error("❌ Model files not found! Make sure 'model_female.h5' and 'model_male.h5' are in the same folder as app.py.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Select Speaker Gender", ["Female", "Male"])
        audio_file = st.file_uploader("Upload Audio", type=['wav'])

    with col2:
        if audio_file is not None:
            st.audio(audio_file)
            
            if st.button("Analyze"):
                with st.spinner("Listening..."):
                    # Save temp file
                    with open("temp.wav", "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                    try:
                        # Process
                        features = process_audio("temp.wav")
                        
                        # Predict
                        if gender == 'Female':
                            preds = model_female.predict(features)
                        else:
                            preds = model_male.predict(features)
                            
                        # Results
                        idx = np.argmax(preds)
                        label = EMOTION_LABELS[idx]
                        conf = np.max(preds) * 100
                        
                        st.success(f"Emotion: **{label.upper()}**")
                        st.info(f"Confidence: {conf:.2f}%")
                        
                        # Chart
                        df = pd.DataFrame({"Emotion": EMOTION_LABELS, "Score": preds[0]})
                        st.bar_chart(df.set_index("Emotion"))
                        
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                    finally:
                        if os.path.exists("temp.wav"):
                            os.remove("temp.wav")
