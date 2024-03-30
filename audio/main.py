import streamlit as st
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        st.error(f"Error encountered while parsing file: {file_path}")
        return None

loaded_model = joblib.load("./svm_model.joblib")

def classify_audio(example_file_path):
    example_features = extract_features(example_file_path)
    if example_features is not None:
        prediction = loaded_model.predict([example_features])
        class_label = "Real" if prediction[0] == 1 else "Fake"
        return f"{class_label} Audio File"
    else:
        return "Error extracting features from the example file."

st.title("deepfake audio check")
st.write("This app classifies audio files as real or fake.")
uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.write("Uploaded file details:")
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    if st.button("Check"):
        result = classify_audio(uploaded_file)
        st.write(result)
