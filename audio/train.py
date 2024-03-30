import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint

data_dir = "../DATASET/TRAINING/audio/data"

# Function to extract additional features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        mel_spec = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        features = np.hstack((mfccs, mel_spec, chroma))
        return features
    except Exception as e:
        print("Error encountered while parsing file:", file_path)
        return None

# Function to load data
def load_data(data_dir):
    fake_files = [os.path.join(data_dir, "fake", f) for f in os.listdir(os.path.join(data_dir, "fake")) if f.endswith(".wav")]
    real_files = [os.path.join(data_dir, "real", f) for f in os.listdir(os.path.join(data_dir, "real")) if f.endswith(".wav")]

    fake_labels = [0] * len(fake_files)
    real_labels = [1] * len(real_files)

    files = fake_files + real_files
    labels = fake_labels + real_labels

    return files, labels

files, labels = load_data(data_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2, random_state=42)

# Extract features
X_train = [extract_features(file) for file in X_train]
X_test = [extract_features(file) for file in X_test]

# Remove None values (in case any files encountered errors)
X_train = [x for x in X_train if x is not None]
X_test = [x for x in X_test if x is not None]

# Data Augmentation: Apply pitch shifting and time stretching
# Add this step after loading the data and before fitting the model

# Define augmentation function
def augment_data(X_train):
    augmented_X_train = []
    for features in X_train:
        # Apply pitch shifting
        pitch_shifted = librosa.effects.pitch_shift(features, sr=22050, n_steps=np.random.uniform(-2, 2))
        # Apply time stretching
        time_stretched = librosa.effects.time_stretch(pitch_shifted, rate=np.random.uniform(0.8, 1.2))
        augmented_X_train.append(time_stretched)
    return augmented_X_train

# Augment training data
X_train_augmented = augment_data(X_train)
X_train.extend(X_train_augmented)
y_train.extend(y_train)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    "n_estimators": randint(100, 1000),
    "max_depth": [None] + list(randint(5, 50).rvs(5)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["auto", "sqrt", "log2"]
}

model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, verbose=2)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
