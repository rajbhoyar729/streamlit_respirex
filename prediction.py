import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Load the disease classes once
DISEASE_CLASSES = pd.read_csv('balanced_patient_data.csv')['disease'].tolist()

class DiseasePredictor:
    """
    Class that loads the model once and reuses it for multiple predictions.
    """
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict(self, mfcc, chroma, mel_spec):
        # Make predictions using the preloaded model
        prediction = self.model.predict({
            "mfcc": mfcc,
            "croma": chroma,
            "mspec": mel_spec
        })
        predicted_class = DISEASE_CLASSES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0]) * 100
        return predicted_class, confidence

# Global instance to reuse the model across predictions
_predictor_instance = None

def get_predictor(model_path):
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DiseasePredictor(model_path)
    return _predictor_instance

def predict_disease(model_path, mfcc, chroma, mel_spec):
    """
    Retrieve the predictor (loading the model only once) and make a prediction.
    """
    predictor = get_predictor(model_path)
    return predictor.predict(mfcc, chroma, mel_spec)
