import streamlit as st
import os
import tempfile
from audio_processing import extract_features
from prediction import predict_disease
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Respiratory Disease Predictor",
    page_icon="🫁",
    layout="centered"
)

def main():
    # Add a title and description
    st.title("Respiratory Disease Prediction System")
    st.markdown("""
    This application analyzes audio recordings of respiratory sounds to predict potential respiratory conditions.
    Upload an audio file to get started.
    """)

    # Path to the trained model
    MODEL_PATH = os.path.join(os.getcwd(), "model", "rd.h5")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Show audio player
        st.audio(uploaded_file)

        # Add a predict button
        if st.button("Predict Disease"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    # Extract features
                    mfcc, chroma, mel_spec = extract_features(temp_path)

                    # Make prediction
                    predicted_class, confidence = predict_disease(MODEL_PATH, mfcc, chroma, mel_spec)

                    # Clean up temporary file
                    os.remove(temp_path)

                    # Display results in a nice format
                    st.success("Analysis Complete!")
                    
                    # Create two columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Condition", predicted_class)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")

                    # Add detailed information
                    st.markdown("### Analysis Details")
                    st.markdown(f"""
                    - **Prediction**: {predicted_class}
                    - **Confidence Score**: {confidence:.2f}%
                    - **Audio Features Analyzed**: 
                        - MFCC (Mel-frequency cepstral coefficients)
                        - Chroma features
                        - Mel-spectrogram
                    """)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.markdown("""
                    Please ensure:
                    - The audio file is in a supported format (WAV or MP3)
                    - The file is not corrupted
                    - The audio contains clear respiratory sounds
                    """)

    # Add information section
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload Audio**: Click the upload button and select your audio file (WAV or MP3 format)
        2. **Review**: Listen to the uploaded audio to ensure it's correct
        3. **Analyze**: Click the 'Predict Disease' button to analyze the respiratory sounds
        4. **Results**: View the predicted condition and confidence score
        
        **Note**: For best results, ensure the audio recording:
        - Is clear and free from background noise
        - Contains distinct respiratory sounds
        - Is recorded in a quiet environment
        """)

if __name__ == "__main__":
    main()