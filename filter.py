import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import librosa
from scipy.signal import butter, lfilter

# Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# Audio preprocessing with bandpass filter
def preprocess_audio(file_path, duration=5, lowcut=100, highcut=1000, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        y_filtered = bandpass_filter(y, lowcut, highcut, sr)
        return y_filtered, sr
    except Exception as e:
        print(f"Error preprocessing audio {file_path}: {e}")
        return None, None

# Feature extraction with preprocessing incorporated
def extract_features(audio_path, sr=22050, duration=5, lowcut=100, highcut=1000):
    y, sr = preprocess_audio(audio_path, duration=duration, lowcut=lowcut, highcut=highcut, sr=sr)
    if y is None:
        return None  # Skip if preprocessing failed

    features = []
    try:
        # Spectral Features
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        
        # Chroma Features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features.extend(chroma)
        
        # MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        features.extend(mfccs)
        
        # Zero Crossing Rate
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Entropy of Energy
        energy = librosa.feature.rms(y=y).flatten()
        energy_entropy = -np.sum(energy * np.log2(energy + 1e-8))
        features.append(energy_entropy)
        
        # Spectral Roll-off
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features.append(spectral_rolloff)

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

    return features

# Data preprocessing and class balancing (remains the same)
def preprocess_and_extract(data_dir, sr=22050, duration=5, lowcut=100, highcut=1000):
    X, y = [], []
    class_counts = {}

    for class_label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_label)
        if os.path.isdir(class_path):
            class_samples = []
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if not file_name.lower().endswith(('.wav', '.mp3')):
                    continue
                try:
                    features = extract_features(file_path, sr, duration, lowcut, highcut)
                    if features:
                        class_samples.append(features)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            if class_samples:
                class_counts[class_label] = len(class_samples)
                X.extend(class_samples)
                y.extend([class_label] * len(class_samples))

    if len(X) == 0 or len(y) == 0:
        print("No valid data found in the dataset directory.")
        return None

    # Balance classes with oversampling
    max_samples = max(class_counts.values())
    balanced_X, balanced_y = [], []

    for class_label in class_counts:
        class_data = [X[i] for i in range(len(y)) if y[i] == class_label]
        class_labels = [class_label] * len(class_data)
        class_data, class_labels = resample(
            class_data, class_labels, replace=True, n_samples=max_samples, random_state=42
        )
        balanced_X.extend(class_data)
        balanced_y.extend(class_labels)

    print(f"Class distribution after balancing: {dict(zip(*np.unique(balanced_y, return_counts=True)))}")
    return np.array(balanced_X), np.array(balanced_y)

# Train model with hyperparameter tuning
def train_classifier(X, y, save_path="lung_sound_classifier.pkl"):
    # Handle NaN values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_

    # Evaluate on training set
    y_train_pred = best_clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training Accuracy:", train_accuracy)

    # Confusion matrix for training data
    train_cm = confusion_matrix(y_train, y_train_pred, labels=best_clf.classes_)
    print("Training Confusion Matrix:")
    print(train_cm)
    ConfusionMatrixDisplay(train_cm, display_labels=best_clf.classes_).plot()

    # Evaluate on testing set
    y_test_pred = best_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Testing Accuracy:", test_accuracy)

    # Confusion matrix for testing data
    test_cm = confusion_matrix(y_test, y_test_pred, labels=best_clf.classes_)
    print("Testing Confusion Matrix:")
    print(test_cm)
    ConfusionMatrixDisplay(test_cm, display_labels=best_clf.classes_).plot()

    print("Best Parameters:", grid_search.best_params_)
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Save the model and scaler to the specified path
    joblib.dump((best_clf, scaler, imputer), save_path)
    print(f"Model and scaler saved at: {save_path}")
    return best_clf

def predict_audio(audio_path, model_path="lung_sound_classifier.pkl", sr=22050):
    # Load the model, scaler, and imputer
    clf, scaler, imputer = joblib.load(model_path)
    
    # Extract features from the audio file
    features = extract_features(audio_path, sr)
    if features:
        # Handle missing values using the imputer
        features_imputed = imputer.transform([features])
        
        # Scale the imputed features
        features_scaled = scaler.transform(features_imputed)
        
        # Get the predicted class
        prediction = clf.predict(features_scaled)[0]
        
        # Get class probabilities
        probabilities = clf.predict_proba(features_scaled)[0]
        class_probabilities = {
            cls: prob for cls, prob in zip(clf.classes_, probabilities)
        }
        
        return prediction, class_probabilities
    else:
        return "Error extracting features.", None


# Main script for training
if _name_ == "_main_":
    dataset_dir = "/kaggle/input/manipulated/HMMMMM/Foldered data"  # Change to your dataset path
    save_model_path = "lung_sound_classifier.pkl"

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
    else:
        result = preprocess_and_extract(dataset_dir)
        if result is None:
            print("No data was loaded. Check your dataset structure and file paths.")
        else:
            X, y = result
            model = train_classifier(X, y, save_path=save_model_path)
            print(f"Model training completed and saved as '{save_model_path}'.")
