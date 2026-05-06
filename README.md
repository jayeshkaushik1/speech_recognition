# Speech Emotion Recognition (SER)

> Deep learning pipeline that classifies human speech into emotion categories using CNN and LSTM architectures.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Two independent models are implemented:

| Model | Notebook | Training Strategy | Classes | Saved As |
|-------|----------|-------------------|---------|----------|
| **CNN** | `cnn/ser_cnn.ipynb` | Combined dataset, all speakers | 7 emotions | `best_model.weights.h5` |
| **LSTM** | `lstm/ser_lstm.ipynb` | Gender-split (male / female) | 6 emotions | `model_male.h5`, `model_female.h5` |

---

## Emotion Classes

| Emotion | LSTM | CNN |
|---------|------|-----|
| neutral | ✓ | ✓ |
| happy | ✓ | ✓ |
| sad | ✓ | ✓ |
| angry | ✓ | ✓ |
| fear | ✓ | ✓ |
| disgust | ✓ | ✓ |
| surprise | — | ✓ |

---

## Project Structure

```
Speech_Emotion_Recognition_Project/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit web app
├── lstm/
│   └── ser_lstm.ipynb              # LSTM pipeline (gender-split, 6 classes)
├── cnn/
│   └── ser_cnn.ipynb               # CNN pipeline (combined, 7 classes)
├── model_male.h5                   # LSTM weights — male voices
├── model_female.h5                 # LSTM weights — female voices
├── best_model.weights.h5           # CNN best checkpoint weights
├── .devcontainer/                  # Docker config (optional)
└── datasets/                       # Sample audio files (if available)
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jayeshkaushik1/speech_recognition.git
cd speech_recognition
```

### 2. Create Virtual Environment (Recommended)
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Option A — Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, upload a `.wav` or `.mp3` file, select model (CNN or LSTM), and get a real-time emotion prediction.

### Option B — Jupyter Notebooks
```bash
# LSTM model
jupyter notebook lstm/ser_lstm.ipynb

# CNN model
jupyter notebook cnn/ser_cnn.ipynb
```
Run all cells in order via **Kernel → Restart & Run All**.

> **Kaggle users:** Add all datasets via **+ Add Data**. Run `os.listdir('/kaggle/input/datasets')` to confirm paths, then update the Dataset Paths cell.

---

## Datasets

| Dataset | Files | Used In | Notes |
|---------|-------|---------|-------|
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 1,440 | LSTM + CNN | 24 actors, male & female |
| [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) | 7,442 | LSTM + CNN | Multimodal, diverse speakers |
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2,800 | LSTM + CNN | Female speakers only |
| [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) | 480 | LSTM + CNN | Male speakers only |
| [EmoDB](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb) | 535 | LSTM only | Berlin emotional speech |

---

## Acoustic Features

Each audio clip is represented using **15 features per time-step**:

- **ZCR** — Zero Crossing Rate (1 feature)
- **RMS Energy** — Root Mean Square Energy (1 feature)
- **MFCCs** — Mel-Frequency Cepstral Coefficients 1–13 (13 features)

Four augmentation variants are extracted per clip (original, noise-augmented, pitch-shifted, pitch-shifted + noise), effectively 4× the dataset size.

---

## Model Architectures

### CNN (`cnn/ser_cnn.ipynb`)

```
Input: (n_features, 1)

Block 1:  Conv1D(512, k=5) → BatchNorm → MaxPool(5, stride=2)
Block 2:  Conv1D(512, k=5) → BatchNorm → MaxPool(5, stride=2) → Dropout(0.2)
Block 3:  Conv1D(256, k=5) → BatchNorm → MaxPool(5, stride=2)
Block 4:  Conv1D(256, k=3) → BatchNorm → MaxPool(5, stride=2) → Dropout(0.2)
Block 5:  Conv1D(128, k=3) → BatchNorm → MaxPool(3, stride=2) → Dropout(0.2)
Head:     Flatten → Dense(512, relu) → BatchNorm → Dense(7, softmax)

Optimizer: Adam  |  Loss: Categorical Cross-Entropy  |  Batch: 64
Max Epochs: 50 (EarlyStopping patience=5, ReduceLROnPlateau patience=3)
Total Parameters: ~7.19M
```

### LSTM (`lstm/ser_lstm.ipynb`)

```
Input: (352 time-steps, 15 features)

LSTM(64, return_sequences=True)
LSTM(64)
Dense(6, softmax)

Optimizer: RMSProp  |  Loss: Categorical Cross-Entropy  |  Batch: 6
Female model: 200 epochs  |  Male model: 95 epochs
Class weighting: balanced (sklearn compute_class_weight)
```

---

## Results

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| CNN (combined) | ~98% | Full classification report in `ser_cnn.ipynb` |
| LSTM — Female | ~88% | |
| LSTM — Male | ~80% | Limited by female-only TESS dataset |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| Model file not found | Ensure `.h5` files are in the project root |
| `FileNotFoundError` (dataset) | Check paths with `os.listdir('/kaggle/input/datasets')` |
| Audio upload fails | Use WAV or MP3, 3–10 seconds recommended |
| Streamlit won't start | Try `streamlit run app.py --logger.level=debug` |
| Out of memory | Reduce batch size or close other apps |
| `NameError: os not defined` | Run all cells from the top — imports cell must execute first |

---

## Future Enhancements

- [ ] Real-time microphone input
- [ ] Multi-language emotion recognition
- [ ] Ensemble CNN + LSTM model
- [ ] Mobile app deployment
- [ ] Emotion intensity measurement
- [ ] Expand male voice training data to improve LSTM male accuracy

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

```
tensorflow>=2.10
librosa>=0.10
numpy
pandas
scikit-learn
streamlit
matplotlib
seaborn
scipy
tqdm
```

---

## Author

**Jayesh Kaushik**  
BTech Student, IIIT Bhopal  
GitHub: [@jayeshkaushik1](https://github.com/jayeshkaushik1)

---

## License

MIT License — free to use for academic and personal projects.
**Last Updated:** November 28, 2025
