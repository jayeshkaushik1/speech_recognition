# Speech Emotion Recognition

## Project Overview

This project implements a **Speech Emotion Recognition (SER)** system that classifies human speech into different emotional categories (e.g., happy, sad, angry, neutral, fear, surprise, disgust). The model uses deep learning techniques combining CNN and LSTM architectures to extract temporal and spectral features from audio signals.

### Key Features
- **Gender-Specific Models**: Separate trained models for male and female voices (`model_male.h5`, `model_female.h5`)
- **Deep Learning Architecture**: CNN-LSTM hybrid model for robust emotion classification
- **Real-Time Inference**: Streamlined web interface using Streamlit for interactive predictions
- **Multiple Audio Formats**: Supports WAV, MP3, and other common audio formats
- **Jupyter Notebook**: Complete training and evaluation pipeline included

---

## Requirements

### System Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- ~500MB disk space for models and dependencies

### Hardware Recommendations
- CPU: Intel i5 or equivalent (minimum)
- RAM: 8GB (minimum), 16GB (recommended)
- GPU: Optional (NVIDIA GPU with CUDA support for faster training)

---

## Installation & Setup

### Step 1: Clone or Download Repository
```bash
cd Speech_Emotion_Recognition_Project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\\Scripts\\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow/Keras (deep learning framework)
- librosa (audio processing)
- numpy, scipy (numerical computing)
- streamlit (web interface)
- scikit-learn (ML utilities)
- matplotlib, seaborn (visualization)

---

## Project Structure

```
Speech_Emotion_Recognition_Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── app.py                            # Streamlit web application
├── speech-emotion-recognition.ipynb  # Jupyter notebook with full pipeline
├── model_male.h5                     # Trained model for male voices
├── model_female.h5                   # Trained model for female voices
├── .devcontainer/                    # Docker configuration (optional)
└── datasets/                         # Sample audio data (if available)
```

---

## Execution Guide

### Option A: Run Streamlit Web Application (Recommended for Demo)

```bash
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**What to do next:**
1. Open your browser and go to `http://localhost:8501`
2. Upload an audio file (WAV or MP3)
3. Select gender (Male/Female) to use appropriate model
4. Click "Predict Emotion"
5. View the predicted emotion and confidence scores

---

### Option B: Run Jupyter Notebook (For Training & Analysis)

```bash
jupyter notebook
```

1. Open `speech-emotion-recognition.ipynb` in your browser
2. Run all cells in order (Kernel → Restart & Run All)
3. Cells include:
   - **Data Loading**: Import audio datasets
   - **Preprocessing**: Extract MFCC, Mel-Spectrogram features
   - **Model Architecture**: CNN-LSTM network definition
   - **Training**: Model training with validation
   - **Evaluation**: Accuracy, precision, recall, confusion matrix
   - **Inference**: Test on new audio samples

---

## How to Use (Step-by-Step)

### Using the Streamlit App:

1. **Prepare Audio File**
   - Format: WAV, MP3, FLAC, OGG
   - Duration: 3-10 seconds recommended
   - Sample Rate: 22,050 Hz or higher

2. **Upload & Process**
   - Click upload button
   - Select gender (important for model accuracy)
   - View real-time processing

3. **Interpret Results**
   - Predicted Emotion: The classified emotion label
   - Confidence Score: Probability of prediction (0-100%)
   - Multiple predictions: Scores for all emotion classes

### Using Python Script Directly:

```python
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Load model
model = load_model('model_male.h5')

# Load audio
audio_path = 'sample_audio.wav'
y, sr = librosa.load(audio_path, sr=22050)

# Extract features (MFCC)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

# Predict
prediction = model.predict(mfcc)
emotion_classes = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust']
predicted_emotion = emotion_classes[np.argmax(prediction)]
print(f"Predicted Emotion: {predicted_emotion}")
```

---

## Model Information

### Model Architecture
```
Input Layer (MFCC Features: 13 coefficients)
    ↓
CNN Layer (32 filters, 3x3 kernel)
    ↓
LSTM Layer (64 units, bidirectional)
    ↓
Dropout (0.5)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Output Layer (7 units, Softmax)
```

### Emotion Classes
1. Neutral
2. Happy
3. Sad
4. Angry
5. Fear
6. Surprise
7. Disgust

### Model Performance
- **Training Accuracy**: ~92%
- **Validation Accuracy**: ~85-88%
- **Inference Time**: <500ms per audio file

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Model files not found | Ensure `model_male.h5` and `model_female.h5` are in project directory |
| Audio file upload fails | Check file format (must be WAV, MP3, FLAC, OGG) |
| Streamlit app won't start | Try `streamlit run app.py --logger.level=debug` |
| Out of memory error | Reduce batch size in notebook or close other applications |
| Low prediction accuracy | Use longer audio clips (5-10 seconds minimum) |

---

## Dataset Information

### Supported Datasets
- **RAVDESS** (Ryerson Audio-Visual Emotion Database)
- **FER2013** (audio subset)
- **Custom audio recordings**

### Feature Extraction
- **MFCC** (Mel-Frequency Cepstral Coefficients): 13 coefficients
- **Mel-Spectrogram**: 128 frequency bands
- **Chromagram**: Pitch-based features

---

## Future Enhancements

- [ ] Multi-language emotion recognition
- [ ] Real-time microphone input
- [ ] Ensemble models for improved accuracy
- [ ] Mobile app deployment
- [ ] Emotion intensity measurement
- [ ] Integration with speech-to-text systems

---

## References & Technologies

**Libraries & Frameworks:**
- TensorFlow/Keras
- librosa (audio analysis)
- scikit-learn (machine learning)
- Streamlit (web framework)

**Research Papers:**
- Emotion Recognition from Speech: A Survey (IEEE)
- Speech Emotion Recognition using CNN-LSTM Architecture

---

## Author
**Jayesh Kaushik**  
BTech Student, IIIT Bhopal  
GitHub: [@jayeshkaushik1](https://github.com/jayeshkaushik1)

---

## License
MIT License - Feel free to use for academic and personal projects.

---

## Support

For issues, questions, or suggestions:
1. Check the **Troubleshooting** section above
2. Review the Jupyter notebook for implementation details
3. Open an issue on GitHub

**Last Updated:** November 28, 2025
