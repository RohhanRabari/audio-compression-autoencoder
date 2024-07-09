import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_to_spectrogram(file_path, n_fft=1024, hop_length=512, n_mels=128):
    # Load audio file
    waveform, sr = librosa.load(file_path, sr=None)
    
    # Compute the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Convert to log scale (dB)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    return log_spectrogram

# Example usage
# log_spectrogram = audio_to_spectrogram('path_to_audio_file.wav')

def preprocess_dataset(data_dir, output_dir, n_fft=1024, hop_length=512, n_mels=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                spectrogram = audio_to_spectrogram(file_path, n_fft, hop_length, n_mels)
                output_file = os.path.join(output_dir, file.replace('.wav', '.npy'))
                np.save(output_file, spectrogram)
                print(f"Processed and saved: {output_file}")

# Example usage
# preprocess_dataset('path_to_dream_sound_dataset', 'path_to_save_spectrograms')

def display_spectrogram(spectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

# Example usage
# spectrogram = np.load('path_to_spectrogram.npy')
# display_spectrogram(spectrogram)

if __name__ == '__main__':
        
    data_dir = '/Users/ronin/AI/audio-compression-autoencoder/data/microsoft-speech-corpus/raw/gu-in-Train'
    output_dir = '/Users/ronin/AI/audio-compression-autoencoder/data/microsoft-speech-corpus/processed/gu-in-Train'

    # preprocess_dataset(data_dir, output_dir)
    
    spectrogram = np.load('/Users/ronin/AI/audio-compression-autoencoder/data/dreamsound/processed/24930__vexst__basic-break.npy')
    display_spectrogram(spectrogram)