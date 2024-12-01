import os
import random
import subprocess
import numpy as np
from pydub import AudioSegment

# NOTE 
# To run, you must use this git repo with the NSNet2 models ...
# https://github.com/microsoft/DNS-Challenge.git
# Baseline modes are in the "NSNet2-baseline" directory 

# I changed the following line in the run_nsnet2.py file
        # INITIAL:  sf.write(outpath.resolve(), outSig, fs)
        # ADJUSTED: sf.write(str(outpath), outSig, fs)

# Also I had to resample the audio files to be at a sampling rate of 48000, rather than 44100 for now

# ADJUST FOR YOUR PERSONAL DIRECTORY
nsnet2_model_path = "C:/Users/B/WPI/Graduate/CS 541 Deep Learning/DNS-Challenge/NSNet2-baseline/nsnet2-20ms-48k-baseline.onnx"
denoised_suffix = "nsnet2-20ms-48k-baseline"

# Where I locally stored the clean / noise audio files 
clean_audio_dir = "./all_audio/clean_audio"
noise_audio_dir = "./all_audio/noisy_audio"

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

def nsnet2(combined_audio_file_path, output_dir="./denoised_audio"):
    """
    Takes in the path to a noisy file and returns the path to 
    the denoised version in the 'denoised_audio' directory.
    """
    denoised_path = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Example command to run the NSNet2 model
    command_list = [
        "python",
        "c:/Users/B/WPI/Graduate/CS 541 Deep Learning/DNS-Challenge/NSNet2-baseline/run_nsnet2.py",
        "-m", nsnet2_model_path,
        "-i", combined_audio_file_path,
        "-o", denoised_path
    ]
    subprocess.run(command_list)

    # Append the suffix to the output file name
    combined_file_name = os.path.basename(combined_audio_file_path)
    denoised_file_name = f"{combined_file_name[:-4]}_{denoised_suffix}.wav"
    return os.path.join(denoised_path, denoised_file_name)

# Main loop for processing files and denoising
sample_rate = 48000
num_processed_files = 10
clean_audio_files = sorted(os.listdir(clean_audio_dir))[:num_processed_files]
noise_audio_files = sorted(os.listdir(noise_audio_dir))[:num_processed_files]
#noise_audio_files = os.listdir(noise_audio_dir)

# Paths of clean, combined, and denoised files
combined_paths = []
denoised_paths = []
clean_paths = []

for clean_file, noise_file in zip(clean_audio_files, noise_audio_files):
    clean_path = os.path.join(clean_audio_dir, clean_file)
    noise_path = os.path.join(noise_audio_dir, noise_file)

    #noise_file = random.choice(noise_audio_files)  # Randomly select a noise file
    #noise_path = os.path.join(noise_audio_dir, noise_file)

    # Load audio files
    clean_audio = AudioSegment.from_file(clean_path)
    noise_audio = AudioSegment.from_file(noise_path)

    # Ensure both audio files have the same frame rate and number of channels
    if clean_audio.frame_rate != noise_audio.frame_rate:
        noise_audio = noise_audio.set_frame_rate(clean_audio.frame_rate)
    if clean_audio.channels != noise_audio.channels:
        noise_audio = noise_audio.set_channels(clean_audio.channels)

    # Truncate or pad to match durations
    min_duration = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_duration]
    noise_audio = noise_audio[:min_duration]

    # Combine the audio files (mix them)
    combined_audio = clean_audio.overlay(noise_audio)

    # Resample to 48000 Hz
    combined_audio = combined_audio.set_frame_rate(48000)

    # Save combined audio for NSNet2 input
    os.makedirs("./combined_audio", exist_ok=True)
    combined_path = f"./combined_audio/{clean_file[0:-4]}-combined_resampled_audio.wav"
    combined_audio.export(combined_path, format="wav")
    combined_paths.append(combined_path)
    clean_paths.append(clean_path)

    print(f"Processing: Clean: {clean_file}, Noise: {noise_file}")
    try:
        denoised_audio_path = nsnet2(combined_path)  # Denoise Audio file  
        denoised_paths.append(denoised_audio_path)
    except Exception as e:
        print(f"Error processing {clean_file} with {noise_file}: {e}")

# Separate loop to calculate SNR
total_snr = 0
for clean_path, denoised_path in zip(clean_paths, denoised_paths):
    try:
        clean_audio = AudioSegment.from_file(clean_path)
        denoised_audio = AudioSegment.from_file(denoised_path)

        snr = calculate_snr(clean_audio, denoised_audio)
        print(f"SNR for {os.path.basename(clean_path)}: {snr:.2f} dB")
        total_snr += snr
    except Exception as e:
        print(f"Error calculating SNR for {clean_path}: {e}")

# Compute the average SNR
average_snr = total_snr / len(denoised_paths) if denoised_paths else 0
print(f"Average SNR across {len(denoised_paths)} samples: {average_snr:.2f} dB")
