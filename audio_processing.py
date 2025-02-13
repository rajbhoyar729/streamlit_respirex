import numpy as np
import librosa
import functools
import concurrent.futures

@functools.lru_cache(maxsize=None)
def load_audio(audio_path, sr=22050):
    """
    Load an audio file with a fixed sampling rate.
    Caching ensures that repeated calls with the same file do not re-read it from disk.
    """
    return librosa.load(audio_path, sr=sr)

def extract_features(audio_path):
    """
    Extract MFCC, Chroma, and Mel-spectrogram features from an audio file using a lower sampling rate and caching.
    """
    # Load the audio file at a lower sampling rate for faster processing
    audio_data, sample_rate = load_audio(audio_path, sr=22050)
    target_frames = 259  # Fixed frame count for consistency

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :target_frames]

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=12)
    if chroma.shape[1] < target_frames:
        chroma = np.pad(chroma, ((0, 0), (0, target_frames - chroma.shape[1])))
    else:
        chroma = chroma[:, :target_frames]

    # Extract Mel-spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    if mel_spec.shape[1] < target_frames:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, target_frames - mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :target_frames]

    # Add batch and channel dimensions
    mfcc = np.expand_dims(np.expand_dims(mfcc, 0), -1)
    chroma = np.expand_dims(np.expand_dims(chroma, 0), -1)
    mel_spec = np.expand_dims(np.expand_dims(mel_spec, 0), -1)

    return mfcc, chroma, mel_spec

def extract_features_parallel(audio_paths, max_workers=4):
    """
    Process a list of audio files in parallel to extract their features.
    Uses ProcessPoolExecutor for concurrent processing.
    """
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(extract_features, path): path for path in audio_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                features = future.result()
                results[path] = features
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return results
