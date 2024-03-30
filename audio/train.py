import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print("Error extracting features from:", file_path)
        return None

data_dir = "../DATASET/TRAINING/audio/data"
fake_files = [os.path.join(data_dir, "fake", f) for f in os.listdir(os.path.join(data_dir, "fake")) if f.endswith(".wav")]
real_files = [os.path.join(data_dir, "real", f) for f in os.listdir(os.path.join(data_dir, "real")) if f.endswith(".wav")]

fake_labels = np.zeros(len(fake_files))
real_labels = np.ones(len(real_files))

files = fake_files + real_files
labels = np.concatenate((fake_labels, real_labels), axis=None)

features = [extract_features(file) for file in files]
features = [f for f in features if f is not None]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("rff  {:.2f}%".format(accuracy * 100))
print("rff report:")
print(classification_report(y_test, y_pred))

rf_model_filename = "rf_model.joblib"
dump(rf_model, rf_model_filename)

svm_model = make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', gamma='scale', random_state=42))
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\n svm: {:.2f}%".format(accuracy_svm * 100))
print("SVm report:")
print(classification_report(y_test, y_pred_svm))

svm_model_filename = "svm_model.joblib"
dump(svm_model, svm_model_filename)

