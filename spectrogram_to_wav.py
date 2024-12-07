import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import istft

# This is only for spectrogram images, the audio is pretty bad though

def spectrogram_to_wav(image_path, output_wav_path, sample_rate=22050, nperseg=512):
    # Step 1: Load the spectrogram image
    img = plt.imread(image_path)
    
    # Convert to grayscale if the image is RGB
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    
    # Rescale pixel values to original spectrogram scale (assuming 0 to 1 range)
    spectrogram = img * 80  # Example scale (adjust if your spectrogram uses a different range)
    
    # Step 2: Inverse log scale if necessary
    spectrogram = 10 ** (spectrogram / 20)  # Assuming dB scale
    
    # Step 3: Create a complex spectrogram
    # This assumes the magnitude-only spectrogram; we need to estimate phase
    angle = np.exp(1j * np.random.uniform(0, 2 * np.pi, spectrogram.shape))  # Random phase
    complex_spectrogram = spectrogram * angle
    
    # Step 4: Invert STFT
    _, reconstructed_audio = istft(complex_spectrogram, nperseg=nperseg)
    
    # Step 5: Normalize the audio
    reconstructed_audio = np.int16(reconstructed_audio / np.max(np.abs(reconstructed_audio)) * 32767)
    
    # Step 6: Save as a WAV file
    write(output_wav_path, sample_rate, reconstructed_audio)

# Example usage
spectrogram_to_wav("E:/CS541 - Deep Learning/noisy_audio_spectrogram/noisy_spectrogram0.png", "E:/CS541 - Deep Learning/SNN-Audio-Denoising/new_spectro.wav")
