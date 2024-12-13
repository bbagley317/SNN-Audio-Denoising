import subprocess
import time
import librosa
import librosa.display
import numpy as np
import noisereduce as nr
import soundfile as sf
import os
from pydub import AudioSegment


sample_rate = 48000
num_processed_files = 20
clean_audio_dir = "./all_audio/clean_audio"
noise_audio_dir = "./all_audio/noisy_audio"
clean_audio_files = os.listdir(clean_audio_dir)[:num_processed_files]  
noise_audio_files = os.listdir(noise_audio_dir)[:num_processed_files]  

nsnet2_model_path = "....../DNS-Challenge/NSNet2-baseline/nsnet2-20ms-48k-baseline.onnx"
denoised_suffix = "nsnet2-20ms-48k-baseline"

os.makedirs("./denoised_ss", exist_ok=True)
os.makedirs("./denoised_nsnet2", exist_ok=True)


# BASELINE MODELS
def spec_subtraction(combined_audio_path):
    '''
    Runs Spectral Subraction using noise reduce library and returns noisy audio path
    '''
    combined_audio, sr = librosa.load(combined_audio_path, sr=sample_rate)  

    # Denoise the combined audio using the noisereduce library

    start_time = time.time()
    denoised_audio = nr.reduce_noise(y=combined_audio, sr=sample_rate)
    end_time = time.time()
    latency = (end_time - start_time) * 1000  

    # Put denoised file in respective directory
    file_name = combined_audio_path.split('/')[-1]
    path = os.path.join("./denoised_ss", file_name)
    sf.write(path, denoised_audio, samplerate=sr)

    # Convert clean and denoised audio into spectrograms Short-Time Fourier Transform (STFT) for evaluation
    # These are complex spectrograms that include both magnitude and phase
    #clean_stft = librosa.stft(clean_audio)
    #noise_stft = librosa.stft(noise_audio)
    #denoised_stft = librosa.stft(denoised_audio)

    return path, latency
    
def nsnet2(combined_audio_file_path, output_dir="./denoised_nsnet2"):
    """
    Takes in the path to a noisy file and returns the path to 
    the denoised version in the 'denoised_audio' directory.
    """
    denoised_path = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Command to run the NSNet2 model
    command_list = [
        "python",
        "c:/Users/B/WPI/Graduate/CS 541 Deep Learning/DNS-Challenge/NSNet2-baseline/run_nsnet2.py",
        "-m", nsnet2_model_path,
        "-i", combined_audio_file_path,
        "-o", denoised_path
    ]

    start_time = time.time()
    subprocess.run(command_list)
    end_time = time.time()
    latency = (end_time - start_time) * 1000  

    # Append the suffix to the output file name
    combined_file_name = os.path.basename(combined_audio_file_path)
    denoised_file_name = f"{combined_file_name[:-4]}_{denoised_suffix}.wav"
    return os.path.join(denoised_path, denoised_file_name), latency

# EVALUATION FUNCTIONS
def calculate_snr(clean_audio, denoised_audio):
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    clean_audio and denoised_audio are both AudioSegment objects.
    """
    clean_samples = np.array(clean_audio.get_array_of_samples(), dtype=float)
    denoised_samples = np.array(denoised_audio.get_array_of_samples(), dtype=float)

    # Ensure arrays are the same length
    min_length = min(len(clean_samples), len(denoised_samples))
    clean_samples = clean_samples[:min_length]
    denoised_samples = denoised_samples[:min_length]

    # Calculate the SNR
    signal_power = np.mean(clean_samples ** 2)
    noise_power = np.mean((clean_samples - denoised_samples) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    return snr

def calculate_si_snr(clean_audio, denoised_audio):
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    clean_audio and denoised_audio are both AudioSegment objects.
    """
    clean_samples = np.array(clean_audio.get_array_of_samples(), dtype=float)
    denoised_samples = np.array(denoised_audio.get_array_of_samples(), dtype=float)

    min_length = min(len(clean_samples), len(denoised_samples))
    clean_samples = clean_samples[:min_length]
    denoised_samples = denoised_samples[:min_length]

    # Remove mean from signals
    clean_samples -= np.mean(clean_samples)
    denoised_samples -= np.mean(denoised_samples)

    # Project denoised signal onto clean signal
    clean_energy = np.dot(clean_samples, clean_samples)
    scaling_factor = np.dot(clean_samples, denoised_samples) / clean_energy
    target_signal = scaling_factor * clean_samples

    # Compute the noise (difference between denoised signal and scaled clean signal)
    noise_signal = denoised_samples - target_signal

    # Compute SI-SNR
    target_power = np.sum(target_signal ** 2)
    noise_power = np.sum(noise_signal ** 2)
    si_snr = 10 * np.log10(target_power / noise_power) if noise_power > 0 else np.inf

    return si_snr

def calculate_spectrogram_error(clean_audio_path, denoised_audio_path):
    """
    Calculate the total error between the clean and denoised spectrograms.
    
    Args:
        clean_audio_path (str): Path to the clean audio file.
        denoised_audio_path (str): Path to the denoised audio file.
    
    Returns:
        float: Total error between the clean and denoised spectrograms.
    """
    # Load the clean and denoised audio
    clean_audio, sr = librosa.load(clean_audio_path, sr=sample_rate)
    denoised_audio, _ = librosa.load(denoised_audio_path, sr=sample_rate)
    
    # Compute the spectrograms (magnitude only)
    clean_spectrogram = np.abs(librosa.stft(clean_audio, n_fft=2048, hop_length=512))
    denoised_spectrogram = np.abs(librosa.stft(denoised_audio, n_fft=2048, hop_length=512))
    
    # Ensure both spectrograms have the same shape
    min_time = min(clean_spectrogram.shape[1], denoised_spectrogram.shape[1])
    clean_spectrogram = clean_spectrogram[:, :min_time]
    denoised_spectrogram = denoised_spectrogram[:, :min_time]
    
    # Calculate the total error (Mean Absolute Error between spectrogram magnitudes)
    error = np.mean(np.abs(clean_spectrogram - denoised_spectrogram))
    
    return error

# Paths of clean, combined, and denoised files
combined_paths = []
clean_paths = []

# Organize processed audio files (clean, noise, and combined)
for clean_file, noise_file in zip(clean_audio_files, noise_audio_files):
    #Load Audio files
    clean_path = os.path.join(clean_audio_dir, clean_file)
    noise_path = os.path.join(noise_audio_dir, noise_file)
    clean_audio = AudioSegment.from_file(clean_path)
    noise_audio = AudioSegment.from_file(noise_path)

    # Ensure both audio files have the same frame rate, number of channels, and duration
    if clean_audio.frame_rate != noise_audio.frame_rate: noise_audio = noise_audio.set_frame_rate(clean_audio.frame_rate)
    if clean_audio.channels != noise_audio.channels: noise_audio = noise_audio.set_channels(clean_audio.channels)
    min_duration = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_duration]
    noise_audio = noise_audio[:min_duration]

    # Combine the audio files / Resample to 48000 Hz 
    combined_audio = clean_audio.overlay(noise_audio)
    combined_audio = combined_audio.set_frame_rate(48000)

    # Save combined audio for NSNet2 input
    os.makedirs("./combined_audio", exist_ok=True)
    combined_path = f"./combined_audio/{clean_file[0:-4]}-combined_resampled_audio.wav"
    combined_audio.export(combined_path, format="wav")
    combined_paths.append(combined_path)
    clean_paths.append(clean_path)

#tot_snr_ss = 0
#tot_snr_nsnet2 = 0
tot_si_snr_ss = 0
tot_si_snr_nsnet2 = 0
tot_error_ss = 0
tot_error_nsnet2 = 0
tot_lat_ss = 0
tot_lat_nsnet2 = 0

for clean_audio_path, combined_audio_path in zip(clean_paths, combined_paths):
    denoised_ss_path, ss_latency = spec_subtraction(combined_audio_path)            # run spec subtraction
    denoised_nsnet2_path, nsnet2_latency = nsnet2(combined_audio_path)             # run nsnet2

    clean_audio = AudioSegment.from_file(clean_audio_path)
    denoised_audio_ss = AudioSegment.from_file(denoised_ss_path)
    denoised_audio_nsnet2 = AudioSegment.from_file(denoised_nsnet2_path)

    # Calculate SNR
    #snr_ss = calculate_snr(clean_audio, denoised_audio_ss)
    #tot_snr_ss += snr_ss
    #print(f"Spectral Subtraction SNR: {snr_ss}")
    #nr_nsnet2 = calculate_snr(clean_audio, denoised_audio_nsnet2)
    #tot_snr_nsnet2 += snr_nsnet2
    #print(f"NSNET2 SNR: {snr_nsnet2}")

    # Calculate SI-SNR
    si_snr_ss = calculate_si_snr(clean_audio, denoised_audio_ss)
    tot_si_snr_ss += si_snr_ss
    print(f"Spectral Subtraction SI-SNR: {si_snr_ss}")

    si_snr_nsnet2 = calculate_si_snr(clean_audio, denoised_audio_nsnet2)
    tot_si_snr_nsnet2 += si_snr_nsnet2
    print(f"NSNET2 SI-SNR: {si_snr_nsnet2}")

    # Calculate spectrogram errors
    error_ss = calculate_spectrogram_error(clean_audio_path, denoised_ss_path)
    tot_error_ss += error_ss
    print(f"Spectrogram Error (Spectral Subtraction): {error_ss}")

    error_nsnet2 = calculate_spectrogram_error(clean_audio_path, denoised_nsnet2_path)
    tot_error_nsnet2 += error_nsnet2
    print(f"Spectrogram Error (NSNET2): {error_nsnet2}")
    print("-"*20)

    # Print Latency times / increment total
    print(f"Latency (ms) (Spectral Subtraction): {ss_latency}")
    tot_lat_ss += ss_latency

    print(f"Latency (ms) (NSNET2):{nsnet2_latency}")
    tot_lat_nsnet2 += nsnet2_latency


# CALCULATE AVERAGES

# Calculate average SNR
#avg_snr_ss = tot_snr_ss / len(clean_paths)
#print(f"Average SNR for Spectral Subtraction: {avg_snr_ss}")
#avg_snr_nsnet2 = tot_snr_nsnet2 / len(clean_paths)
#print(f"Average SNR for Spectral Subtraction: {avg_snr_nsnet2}")

# Calculate average SNR
avg_si_snr_ss = tot_si_snr_ss / len(clean_paths)
print(f"Average SI-SNR for Spectral Subtraction: {avg_si_snr_ss}")

avg_si_snr_nsnet2 = tot_si_snr_nsnet2 / len(clean_paths)
print(f"Average SI-SNR for NSNET2: {avg_si_snr_nsnet2}")

# Calculate average spectrogram error
avg_error_ss = tot_error_ss / len(clean_paths)
print(f"Average Spectrogram Error for Spectral Subtraction: {avg_error_ss}")

avg_error_nsnet2 = tot_error_nsnet2 / len(clean_paths)
print(f"Average Spectrogram Error for NSNET2: {avg_error_nsnet2}")

# Calculate average latency
avg_lat_ss = tot_lat_ss / len(clean_paths)
print(f"Average Latency for Spectral Subtraction: {avg_lat_ss}")

avg_lat_nsnet2 = tot_lat_nsnet2 / len(clean_paths)
print(f"Average Latency for NSNET2: {avg_lat_nsnet2}")