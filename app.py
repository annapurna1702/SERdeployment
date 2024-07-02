import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import joblib
import tempfile

np.import_array()

svm_model = joblib.load("SVMexec_modeltesting.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(audio_data, sr):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)

        features = np.hstack([
            np.mean(mfccs, axis=1), np.var(mfccs, axis=1),
            np.mean(chroma, axis=1), np.var(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.var(spectral_contrast, axis=1)
        ])
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def predict_emotion(audio_data, sr):
    features = extract_features(audio_data, sr)
    if features is not None:
        features = scaler.transform([features])
        prediction = svm_model.predict(features)
        return prediction[0]
    else:
        return None

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Me"])

if page == "Home":
    st.title("Speech Emotion Recognition for Malayalam Male Voices")
    st.write("This web app processes Malayalam audio files in .wav format, featuring Malayalam male voices. It classifies emotions as sad and not sad.")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                temp_audio_file.write(uploaded_file.getbuffer())
                temp_audio_file_path = temp_audio_file.name

            audio_data, sr = librosa.load(temp_audio_file_path, sr=None)

            emotion = predict_emotion(audio_data, sr)
            st.audio(uploaded_file)

            if emotion is not None:
                if emotion == 'notsad':
                    st.success(f"The predicted emotion is: Not sad")
                else:
                    st.success(f"The predicted emotion is: Sad")
            else:
                st.warning("Unable to predict emotion.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

elif page == "About Me":
    st.title("About Me")
    st.write("Hi! I am Annapurna Padmanabhan, a post-graduate student in Digital University, Kerala, India.")
    st.write("This app was developed as part of my team project in which we trained various models.")
    st.write("Feel free to contact me:")
    st.markdown("[GitHub](https://github.com/annapurna1702)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/annapurnapadmanabhan/)")
