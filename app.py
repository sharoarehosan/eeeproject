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

# Initialize Flask app
app = Flask(__name__)

# Firebase configurationpytho
#cred = credentials.Certificate('path/to/new/firebase.json')

# Firebase configuration
#firebase_admin.initialize_app(cred, {'storageBucket': 'lungsoundeee.firebasestorage.app'})

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
            #bucket = storage.bucket()
            #blob = bucket.blob(filename)
            #blob.upload_from_filename(filepath)

           # print(f"File path: {filepath}")
           # print(f"Firebase bucket: {firebase_admin.get_app().name}")




            # Process audio to get length
            try:
                audio_prediction = predict(filepath)
                prediction = audio_prediction[:-3].upper()
            except Exception as e:
                print(f"Error processing audio: {e}")
                return str(e)


            # Delete the local file after upload
            os.remove(filepath)

            return render_template('index.html', audio_length=prediction)
        return redirect(url_for('index'))
    except Exception as e:
        return str(e)  # Return error message for debugging

# Function to get audio length
def features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        
        # Extract Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # Extract Chroma Feature
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma)
        
        # Extract MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Extract Root Mean Square (RMS) Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        
        # Return all features as a single array
        return [zcr_mean, spectral_centroid_mean, chroma_mean] + list(mfcc_mean) + [rms_mean]
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def predict(file_path):
    # Load model, encoder, and scaler
    with open("lung_sound_svm_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("label_encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Extract features
    feature = features(file_path)
    if feature is None:
        return "Error extracting features from audio."
    
    # Standardize features
    feature = scaler.transform([feature])
    
    # Predict
    prediction = model.predict(feature)
    class_name = encoder.inverse_transform(prediction)
    return class_name[0]

if __name__ == '__main__':
    app.run(debug=True)
