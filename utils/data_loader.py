import os
import numpy as np
from pathlib import Path

# def load_spectrograms(data_dir):
#     spectrograms = []
#     max_shape = [0, 0]

#     # First pass: determine the maximum shape
#     for file_path in Path(data_dir).rglob('*.npy'):
#         spectrogram = np.load(file_path)
#         max_shape[0] = max(max_shape[0], spectrogram.shape[0])
#         max_shape[1] = max(max_shape[1], spectrogram.shape[1])
#         spectrograms.append(spectrogram)

#     # Second pass: pad all spectrograms to the maximum shape
#     padded_spectrograms = []
#     for spectrogram in spectrograms:
#         padded_spectrogram = pad_to_shape(spectrogram, max_shape)
#         padded_spectrograms.append(padded_spectrogram)
    
#     # Convert list to a NumPy array and expand dimensions
#     padded_spectrograms = np.array(padded_spectrograms)
#     padded_spectrograms = np.expand_dims(padded_spectrograms, axis=-1)
    
#     return padded_spectrograms

# def pad_to_shape(spectrogram, target_shape):
#     padding = [
#         (0, target_shape[0] - spectrogram.shape[0]),  # Padding for the first dimension
#         (0, target_shape[1] - spectrogram.shape[1])   # Padding for the second dimension
#     ]
#     padded_spectrogram = np.pad(spectrogram, padding, mode='constant')
#     return padded_spectrogram

def load_and_pad_spectrogram(file_path, target_shape=(128, 728)):
    spectrogram = np.load(file_path)
    
    print(f'Original shape: {np.shape(spectrogram)}')
    # Calculate padding widths
    pad_width = ((0, max(0, target_shape[0] - spectrogram.shape[0])), 
                 (0, max(0, target_shape[1] - spectrogram.shape[1])))
    
    print(f'padwith: {pad_width}')
    
    # Pad spectrogram to target shape
    padded_spectrogram = np.pad(spectrogram, pad_width, mode='constant')
    
    print(f'padded spectogram shape: {np.shape(padded_spectrogram)}')
    # Ensure the shape matches target shape exactly
    padded_spectrogram = padded_spectrogram[:target_shape[0], :target_shape[1]]
    
    return padded_spectrogram

def load_spectrograms(processed_data_path, target_shape=(128, 728)):
    spectrograms = []
    for filename in os.listdir(processed_data_path):
        if filename.endswith('.npy'):
            print(filename)
            file_path = os.path.join(processed_data_path, filename)
            # print(' and pad called')
            spectrogram = load_and_pad_spectrogram(file_path, target_shape)
            spectrograms.append(spectrogram)
            print('spectogram added')
    
    # Add channel dimension and convert to numpy array
    spectrograms = np.expand_dims(np.array(spectrograms), axis=-1)
    return spectrograms

import os
import numpy as np

def load_spectrograms_and_find_shape(processed_data_path):
    spectrograms = []
    min_shape = None
    max_shape = None
    
    for filename in os.listdir(processed_data_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(processed_data_path, filename)
            spectrogram = np.load(file_path)
            
            # Append the spectrogram to the list
            spectrograms.append(spectrogram)
            
            # Get the shape of the spectrogram
            shape = np.shape(spectrogram)
            
            # Initialize min_shape and max_shape if they are None
            if min_shape is None and max_shape is None:
                min_shape = shape
                max_shape = shape
            else:
                # Update min and max shapes
                min_shape = (min(min_shape[0], shape[0]), min(min_shape[1], shape[1]))
                max_shape = (max(max_shape[0], shape[0]), max(max_shape[1], shape[1]))
    
    return spectrograms, min_shape, max_shape

if __name__ == '__main__':
    
    spectrograms, smin, smax = load_spectrograms_and_find_shape('/Users/ronin/AI/audio-compression-autoencoder/data/microsoft-speech-corpus/processed/gu-in-Train')
    # print(np.shape(spectrograms))
    print(f"Minimum shape: {smin}")
    print(f"Maximum shape: {smax}")