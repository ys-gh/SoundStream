import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from dataset import NSynthDataset
from loss import (adversarial_d_loss, adversarial_g_loss, feature_loss,
                  spectral_reconstruction_loss)
from net import SoundStream, STFTDiscriminator, WaveDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
N_EPOCHS = 2
BATCH_SIZE = 4

soundstream = SoundStream(C=1, D=1, n_q=1, codebook_size=1)
wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
W, H = 1024, 256
stft_disc = STFTDiscriminator(C=1, F_bins=W//2)

soundstream.to(device)
wave_disc.to(device)
stft_disc.to(device)

def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths


train_dataset = NSynthDataset(audio_dir="./toy/train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=2)
sr = train_dataset.sr

valid_dataset = NSynthDataset(audio_dir="./toy/valid")
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=2)

test_dataset = NSynthDataset(audio_dir="./toy/test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=2)


optimizer_g = optim.Adam(soundstream.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

def criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft, sr, dev):
    adv_g = LAMBDA_ADV*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave)
    f_loss = LAMBDA_FEAT*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
    spec_loss = LAMBDA_REC*spectral_reconstruction_loss(x, G_x, sr, dev)
    return  adv_g + f_loss + spec_loss


criterion_d = adversarial_d_loss

best_model = soundstream.state_dict().copy()
best_val_loss = float("inf")

history = {
    "train": {"d": [], "g": []},
    "valid": {"d": [], "g": []},
    "test": {"d": [], "g": []}
}

for epoch in range(1, N_EPOCHS+1):
    
    soundstream.train()
    stft_disc.train()
    wave_disc.train()
    
    train_loss_d = 0.0
    train_loss_g = 0.0
    for x, lengths_x in tqdm(train_loader):
        x = x.to(device)
        lengths_x = lengths_x.to(device)
        
        G_x = soundstream(x)
        
        s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
        
        lengths_stft = stft_disc.features_lengths(lengths_s_x)
        lengths_wave = wave_disc.features_lengths(lengths_x)
        
        features_stft_disc_x = stft_disc(s_x)
        features_wave_disc_x = wave_disc(x)
        
        features_stft_disc_G_x = stft_disc(s_G_x)
        features_wave_disc_G_x = wave_disc(G_x)
        
        loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
        train_loss_g += loss_g.item()
        
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        
        features_stft_disc_x = stft_disc(s_x)
        features_wave_disc_x = wave_disc(x)
        
        features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = wave_disc(G_x.detach())
        
        loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
        
        train_loss_d += loss_d.item()
        
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
    
    history["train"]["d"].append(train_loss_d/len(train_loader))
    history["train"]["g"].append(train_loss_g/len(train_loader))
    
    with torch.no_grad():
        stft_disc.eval()
        wave_disc.eval()
        
        valid_loss_d = 0.0
        valid_loss_g = 0.0
        for x, lengths_x in tqdm(valid_loader):
            x = x.to(device)
            lengths_x = lengths_x.to(device)
        
            G_x = soundstream(x)
            
            s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
            s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
            
            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x = stft_disc(s_G_x)
            features_wave_disc_G_x = wave_disc(G_x)
            
            loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
            valid_loss_g += loss_g.item()
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())
            
            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
            
            valid_loss_d += loss_d.item()
        
        if valid_loss_g < best_val_loss:
            best_model = soundstream.state_dict().copy()
            best_val_loss = valid_loss_g
        
        history["valid"]["d"].append(valid_loss_d/len(valid_loader))
        history["valid"]["g"].append(valid_loss_g/len(valid_loader))
    
    with torch.no_grad():
        stft_disc.eval()
        wave_disc.eval()
        
        test_loss_d = 0.0
        test_loss_g = 0.0
        for x, lengths_x in tqdm(test_loader):
            x = x.to(device)
            lengths_x = lengths_x.to(device)
        
            G_x = soundstream(x)
            
            s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
            s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
            
            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x = stft_disc(s_G_x)
            features_wave_disc_G_x = wave_disc(G_x)
            
            loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
            test_loss_g += loss_g.item()
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())
            
            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
            
            test_loss_d += loss_d.item()

            # save audio
            for i in range(G_x.size(0)):
                output_file = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}.wav")
                torchaudio.save(output_file, G_x[i].detach().cpu(), sr)
        
        history["test"]["d"].append(test_loss_d/len(test_loader))
        history["test"]["g"].append(test_loss_g/len(test_loader))
            
            
        