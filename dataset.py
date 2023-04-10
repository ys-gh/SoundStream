import glob
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir, trim_length=1):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.wav")
        self.audio_dir = audio_dir
        _, self.sr = torchaudio.load(self.filenames[0])
        self.trim_length = trim_length
        
    def __len__(self):
        return len(self.filenames)
    

    def __getitem__(self, index):
        file_path = self.filenames[index]
        # file_path = os.path.join(self.audio_dir, filename)
        waveform, _ = torchaudio.load(file_path)

        # 指定の長さにトリミングする
        target_length = self.sr * self.trim_length
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.cat((waveform, torch.zeros(1, padding)), dim=-1)

        return waveform
