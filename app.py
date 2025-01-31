import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

# Load machine learning model and scaler
svm_model = joblib.load("SVMexec_modeltesting.pkl")
scaler = joblib.load("scaler.pkl")

# Function to extract features from audio data
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

# Function to predict emotion from audio data
def predict_emotion(audio_data, sr):
    features = extract_features(audio_data, sr)
    if features is not None:
        features = scaler.transform([features])
        prediction = svm_model.predict(features)
        return prediction[0]
    else:
        return None

# Set custom CSS style
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Me"])

# Main content based on sidebar selection
if page == "Home":
    st.title("Speech Emotion Recognition for Malayalam Male Voices")
    st.write("This web app processes Malayalam audio files in .wav format, featuring Malayalam male voices. It classifies emotions as sad and not sad.")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav","mp3"])

    if uploaded_file is not None:
        try:
            # Use a temporary file to avoid sampling issues
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                temp_audio_file.write(uploaded_file.getbuffer())
                temp_audio_file_path = temp_audio_file.name

            # Load audio data and sample rate using librosa
            audio_data, sr = librosa.load(temp_audio_file_path, sr=None)

            # Predict emotion
            emotion = predict_emotion(audio_data, sr)
            st.audio(uploaded_file)

            # Display predicted emotion
            if emotion is not None:
                if emotion == 'sad':
                    st.success(f"The predicted emotion is: Sad")
                else:
                    st.success(f"The predicted emotion is: Not Sad")
            else:
                st.warning("Unable to predict emotion.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

elif page == "About Me":
    st.title("About Me")
    st.write("Hi! I am Annapurna Padmanabhan, a post-graduate student in Digital University, Kerala, India.")
    st.write("This app was developed on the model that I trained as part of my team project in which the team trained various models.")
    st.write("Feel free to contact me:")
    st.markdown("[GitHub](https://github.com/annapurna1702)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/annapurnapadmanabhan/)")
