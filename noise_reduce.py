import librosa
import librosa.display
import numpy as np
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
import os
import random 

sample_rate = 44100
clean_audio_dir = "./all_audio/clean_audio"
noise_audio_dir = "./all_audio/noisy_audio"
num_processed_files = 150
clean_audio_files = sorted(os.listdir(clean_audio_dir))[:num_processed_files]  
noise_audio_files = os.listdir(noise_audio_dir) 

def spec_subtraction(clean_audio_path, noise_audio_path, noise_scaling_factor=0.02):
    # Load the clean and noise audio files 
    clean_audio, sr_clean = librosa.load(clean_audio_path, sr=sample_rate)  
    noise_audio, sr_noise = librosa.load(noise_audio_path, sr=sample_rate) 

    # Truncate the longer audio file to match the shorter one
    min_length = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_length]
    noise_audio = noise_audio[:min_length]
    noise_audio *= noise_scaling_factor

    # Create combined audio file
    combined_audio = clean_audio + noise_audio 
    #sf.write("combined_audio.wav", combined_audio, sample_rate)

    # Denoise the combined audio using the noisereduce library
    denoised_audio = nr.reduce_noise(y=combined_audio, sr=sample_rate)
    #sf.write("denoised_audio.wav", denoised_audio, sample_rate)

    # Convert clean and denoised audio into spectrograms Short-Time Fourier Transform (STFT) for evaluation
    # These are complex spectrograms that include both magnitude and phase
    clean_stft = librosa.stft(clean_audio)
    noise_stft = librosa.stft(noise_audio)
    denoised_stft = librosa.stft(denoised_audio)

    # Seperate complex STFT into magnitude and phase
    clean_magnitude, clean_phase = librosa.magphase(clean_stft)
    noise_magnitude, noise_phase = librosa.magphase(noise_stft)
    denoised_magnitude, denoised_phase = librosa.magphase(denoised_stft)

    # Calculate the Signal-to-Noise Ratio (SNR)
    noise = clean_magnitude - denoised_magnitude

    # Compute the Signal-to-Noise Ratio (SNR) in dB
    signal_power = np.sum(clean_magnitude**2)
    noise_power = np.sum(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)

    print(f"SNR (in dB) of the denoised audio compared to the clean audio: {snr:.2f} dB")
    
    # Plot the spectrograms for visualization    
    '''
        plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(clean_magnitude, ref=np.max), x_axis='time', y_axis='log')
    plt.title("Clean Audio Spectrogram")

    plt.subplot(2, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(noise_magnitude, ref=np.max), x_axis='time', y_axis='log')
    plt.title("Noisy Audio Spectrogram")

    plt.subplot(2, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(denoised_magnitude, ref=np.max), x_axis='time', y_axis='log')
    plt.title("Denoised Audio Spectrogram")

    plt.tight_layout()
    plt.show()
    '''

    return snr
    
# Runs spectral subtraction with noisereduce on the first "num_processed_files" of clean audio samples
# Randomly chooses a noise sample for each of the clean audio samples
# Calculates average Signal to Noise Ratio
# Commented out lines in spec_subtraction function to visualize spectrograms and output denoised audio files
total_snr = 0
for clean_file in clean_audio_files:
    clean_path = os.path.join(clean_audio_dir, clean_file)
    noise_file = random.choice(noise_audio_files)  # Randomly select a noise file
    noise_path = os.path.join(noise_audio_dir, noise_file)

    print(f"Processing: Clean: {clean_file}, Noise: {noise_file}")
    snr = spec_subtraction(clean_path, noise_path)
    total_snr += snr

# Compute the average SNR
average_snr = total_snr / len(clean_audio_files)
print(f"Average SNR across {len(clean_audio_files)} samples: {average_snr:.2f} dB")





