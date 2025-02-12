# Respiratory Disease Prediction System

A machine learning-based web application that analyzes respiratory sound recordings to predict potential respiratory conditions. The system uses audio processing techniques and deep learning to extract features and make predictions.

## 🌟 Features

- Real-time audio analysis
- Multiple feature extraction (MFCC, Chroma, Mel-spectrogram)
- User-friendly web interface using Streamlit
- FastAPI backend for efficient processing
- CPU-optimized performance
- Detailed prediction results with confidence scores

## 🔧 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Audio files in WAV or MP3 format

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/respiratory-disease-predictor.git
cd respiratory-disease-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model:
Place the trained model file (`rd.h5`) in the `model` directory.

## 🚀 Usage

### Running the Streamlit App

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an audio file and click "Predict Disease" to get results

### Using the FastAPI Backend

1. Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. Send POST requests to `http://localhost:8000/predict/` with audio files
