import glob
import os
import re

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from pystoi import stoi

import sys 
sys.path.append(os.path.join(os.getcwd(), "spiking-fullsubnet"))

from audiozen.acoustics.io import subsample

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility


# takes clean and noisy data and generates data in batches
class DNSAudio(Dataset):
    def __init__(self, noisy_dir, clean_dir, length, limit=None, offset=0, sublen=6, train=True) -> None:
        """Audio dataset loader for DNS.
        Args:
            root: Path of the dataset location, by default './'.
        """
        super().__init__()
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.len = length
        self.sublen = sublen
        self.train = train

    
    def __len__(self) -> int:
        """Length of the dataset."""
        return self.len

    def __getitem__(self, n):
        """Gets the nth sample from the dataset.
        Args:
            n: Index of the sample to be retrieved.
        Returns:
            Noisy audio sample, clean audio sample, noise audio sample, sample metadata.
        """

        metadata = {}

        noisy_audio, sampling_frequency = sf.read(os.path.join(self.noisy_dir, f'noisy_audio{n}.wav'))
        clean_audio, _ = sf.read(os.path.join(self.clean_dir, f'clean_audio{n}.wav'))

        #num_samples = 30 * sampling_frequency  # 30 sec data
        num_samples = 10 * sampling_frequency  # 10 sec data
        
        train_num_samples = self.sublen * sampling_frequency
        metadata["fs"] = sampling_frequency
        
        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate([noisy_audio, np.zeros(num_samples - len(noisy_audio))])
        
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio, np.zeros(num_samples - len(clean_audio))])

        noisy_audio = noisy_audio.astype(np.float32)
        clean_audio = clean_audio.astype(np.float32)

        if self.train:
            noisy_audio, start_position = subsample(
                noisy_audio,
                subsample_length=train_num_samples,
                return_start_idx=True,
            )
            clean_audio = subsample(
                clean_audio,
                subsample_length=train_num_samples,
                start_idx=start_position,
            )

        return noisy_audio, clean_audio, metadata


if __name__ == "__main__":
    print(os.getcwd())
    train_set = DNSAudio(noisy_dir=os.path.join(os.getcwd(),"noisy_audio\\noisy_audio"), clean_dir=os.path.join(os.getcwd(),"clean_audio\\clean_audio"),length=8000)

    # taining set
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

    print(next(iter(train_dataloader))[0].shape)
    print(next(iter(train_dataloader))[1].shape)
    print(next(iter(train_dataloader))[2])
    
    preds, target, metadata = next(iter(train_dataloader))
    
    print(preds.shape)
    print(target.shape)
    print(metadata)
    d = stoi(target[0].ravel(), preds[0].ravel(), 22050, extended=False)
    print(d)
