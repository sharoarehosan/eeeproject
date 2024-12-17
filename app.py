import os
import firebase_admin
from firebase_admin import credentials, storage
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pydub.utils import mediainfo
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import joblib
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)

# Firebase configurationpytho
cred = credentials.Certificate('firebase.json')
# Firebase configuration
firebase_admin.initialize_app(cred, {'storageBucket': 'lungsoundeee.firebasestorage.app'})

#Loading Model
# File upload configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Upload file to Firebase Storage
            # bucket = storage.bucket()
            # blob = bucket.blob(filename)
            # blob.upload_from_filename(filepath)

            # print(f"File path: {filepath}")
            # print(f"Firebase bucket: {firebase_admin.get_app().name}")



            filpath = cut_middle(filepath)
            # Process audio to get length
            try:
                prediction, probabilities = predict_audio(filpath)
                #prediction = audio_prediction[:-3].upper()
            except Exception as e:
                print(f"Error processing audio: {e}")
                return str(e)


            # Delete the local file after upload
            os.remove(filepath)

            return render_template('index.html', audio_length=prediction,probabilities=probabilities)
        return redirect(url_for('index'))
    except Exception as e:
        return str(e)  # Return error message for debugging
    
def predict_audio(audio_path, model_path="lung_sound_classifier.pkl", sr=22050):
    # Load the model, scaler, and imputer
    clf, scaler, imputer = joblib.load(model_path)
    
    # Extract features from the audio file
    feats = features(audio_path, sr)
    if feats:
        # Handle missing values using the imputer
        features_imputed = imputer.transform([feats])
        
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
    
# Function to get audio length
def features(audio_path, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr)
    y = librosa.effects.preemphasis(y)  # Pre-emphasis for noise reduction
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

def cut_middle(input_file_path):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(input_file_path)
    
    # Get the duration of the audio in milliseconds
    duration = len(audio)
    
    # Check if the duration is greater than 10 seconds (10000 ms)
    if duration > 10000:
        # Calculate the middle section to remove (middle 10 seconds)
        middle_start = duration // 2 - 5000  # Start 5 seconds before the middle
        middle_end = middle_start + 10000  # End 5 seconds after the middle
        
        # Cut the middle part out
        audio = audio[:middle_start] + audio[middle_end:]
    
    # Create a new file path for the trimmed audio file
    output_file_path = input_file_path.replace(".wav", "_trimmed.wav")
    
    # Export the modified audio to the new file (in WAV format)
    audio.export(output_file_path, format="wav")
    
    # Return the path of the modified audio
    return output_file_path

if __name__ == '__main__':
    app.run(debug=True)
