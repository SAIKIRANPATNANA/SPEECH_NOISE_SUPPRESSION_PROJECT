"""
dataset.py - Paired Speech and Noise PyTorch Dataset
====================================================

A CSE Guide to Audio DataLoaders & Masking:
------------------------------------------
1. Why On-The-Fly Mixing?
   Instead of pre-rendering millions of gigabytes of noisy audio files to disk,
   we mix clean speech + noise dynamically inside __getitem__ at random SNR levels.
   This provides INFINITE DATA AUGMENTATION: the model never sees the exact same
   noisy audio twice!

2. What is an Ideal Ratio Mask (IRM)?
   In computer graphics, alpha blending combines foreground and background:
     C = alpha * F + (1 - alpha) * B.
   In audio AI, an Ideal Ratio Mask is an acoustic alpha channel!
   Formula:
     IRM[f, t] = |S[f, t]| / ( |S[f, t]| + |N[f, t]| + eps )
   - If a pixel is 100% voice: |S| >> |N|  => IRM ~ 1.0 (Keep untouched)
   - If a pixel is 100% noise: |N| >> |S|  => IRM ~ 0.0 (Silence it)
   - If voice and noise are equal:          => IRM = 0.5 (Attenuate by 6 dB)

3. Tensor Shapes:
   - Audio segment: 32,768 samples (2.048s @ 16 kHz)
   - STFT with n_fft=512, hop=128 gives 257 frequency bins x 257 time frames
   - We crop to [1, 256, 256] (1 channel, 256 frequency bins, 256 time frames)
   - Perfect power-of-2 dimensions for 2D Convolutions!
"""

import os
import glob
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

from src.audio_processing import compute_stft, pad_or_crop_spectrogram
from src.noise_generator import mix_at_snr


class SpeechNoiseDataset(Dataset):
    """
    PyTorch Dataset that dynamically mixes clean speech with noise audio
    at random SNR levels and computes STFT magnitude, phase, and target IRM masks.
    """

    def __init__(
        self,
        clean_audio_paths: List[str],
        noise_audio_paths: List[str],
        segment_samples: int = 32768,      # ~2.048 seconds @ 16kHz
        sr: int = 16000,
        snr_range: Tuple[float, float] = (-5.0, 15.0),
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        target_spec_dim: int = 256,
        is_validation: bool = False
    ):
        super().__init__()
        self.clean_paths = sorted(clean_audio_paths)
        self.noise_paths = sorted(noise_audio_paths)
        self.segment_samples = segment_samples
        self.sr = sr
        self.snr_min, self.snr_max = snr_range
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_spec_dim = target_spec_dim
        self.is_validation = is_validation

        if not self.clean_paths:
            raise ValueError("No clean audio paths provided.")
        if not self.noise_paths:
            raise ValueError("No noise audio paths provided.")

        # Pre-load noise files into memory for fast batch generation
        self.noise_buffers = []
        for path in self.noise_paths:
            try:
                audio, _ = librosa.load(path, sr=self.sr, mono=True)
                if len(audio) > 0:
                    self.noise_buffers.append(audio.astype(np.float32))
            except Exception as e:
                print(f"Warning: Failed to load noise file {path}: {e}")

        if not self.noise_buffers:
            raise ValueError("Failed to load any valid noise audio files.")

    def __len__(self) -> int:
        return len(self.clean_paths)

    def _get_segment(self, audio: np.ndarray) -> np.ndarray:
        """Crop a random segment or pad with zeros if audio is shorter."""
        length = len(audio)
        if length >= self.segment_samples:
            if self.is_validation:
                # Deterministic center crop for validation
                start = (length - self.segment_samples) // 2
            else:
                # Random crop for training data augmentation
                start = np.random.randint(0, length - self.segment_samples + 1)
            return audio[start:start + self.segment_samples]
        else:
            # Pad with zeros
            pad_needed = self.segment_samples - length
            return np.pad(audio, (0, pad_needed), mode='constant')

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean_path = self.clean_paths[idx]
        
        # 1. Load clean speech
        clean_audio, _ = librosa.load(clean_path, sr=self.sr, mono=True)
        clean_segment = self._get_segment(clean_audio)

        # Normalize clean speech segment
        max_clean = np.max(np.abs(clean_segment))
        if max_clean > 1e-6:
            clean_segment = clean_segment / max_clean

        # 2. Pick a noise source
        if self.is_validation:
            # Deterministic noise selection for validation
            noise_idx = idx % len(self.noise_buffers)
            target_snr = self.snr_min + ((self.snr_max - self.snr_min) * (idx / max(len(self), 1)))
        else:
            noise_idx = np.random.randint(0, len(self.noise_buffers))
            target_snr = float(np.random.uniform(self.snr_min, self.snr_max))

        noise_audio = self.noise_buffers[noise_idx]
        noise_segment = self._get_segment(noise_audio)

        # 3. Mix at target SNR
        noisy_segment, scaled_noise, actual_snr = mix_at_snr(
            clean_segment, noise_segment, target_snr_db=target_snr
        )

        # 4. Compute STFTs
        clean_mag, _ = compute_stft(clean_segment, self.n_fft, self.hop_length, self.win_length)
        noise_mag, _ = compute_stft(scaled_noise, self.n_fft, self.hop_length, self.win_length)
        noisy_mag, noisy_phase = compute_stft(noisy_segment, self.n_fft, self.hop_length, self.win_length)

        # 5. Crop / Pad spectrograms to uniform [256, 256] power-of-2 dimensions
        clean_mag_crop = pad_or_crop_spectrogram(clean_mag, self.target_spec_dim, self.target_spec_dim)
        noisy_mag_crop = pad_or_crop_spectrogram(noisy_mag, self.target_spec_dim, self.target_spec_dim)
        noise_mag_crop = pad_or_crop_spectrogram(noise_mag, self.target_spec_dim, self.target_spec_dim)
        noisy_phase_crop = pad_or_crop_spectrogram(noisy_phase, self.target_spec_dim, self.target_spec_dim)

        # 6. Compute Ideal Ratio Mask (IRM)
        # Formula: IRM = |S| / ( |S| + |N| + eps )
        eps = 1e-8
        irm_mask = clean_mag_crop / (clean_mag_crop + noise_mag_crop + eps)
        irm_mask = np.clip(irm_mask, 0.0, 1.0)

        # 7. Convert to PyTorch tensors with [channel, freq, time] shape (channel=1)
        return {
            "noisy_mag": torch.from_numpy(noisy_mag_crop).unsqueeze(0),       # [1, 256, 256]
            "clean_mag": torch.from_numpy(clean_mag_crop).unsqueeze(0),       # [1, 256, 256]
            "irm_mask": torch.from_numpy(irm_mask).unsqueeze(0),             # [1, 256, 256]
            "noisy_phase": torch.from_numpy(noisy_phase_crop).unsqueeze(0),   # [1, 256, 256]
            "snr_db": torch.tensor(actual_snr, dtype=torch.float32),
            "file_name": os.path.basename(clean_path)
        }


def create_dataloaders(
    clean_dir: str,
    noise_dir: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 2,
    snr_range: Tuple[float, float] = (-5.0, 15.0),
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Split clean speech into train and validation sets, and build PyTorch DataLoaders.
    """
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))

    if not clean_files:
        raise FileNotFoundError(f"No WAV files found in clean_dir: {clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No WAV files found in noise_dir: {noise_dir}")

    np.random.seed(random_seed)
    indices = np.random.permutation(len(clean_files))
    split_point = int(len(clean_files) * (1.0 - val_split))

    train_files = [clean_files[i] for i in indices[:split_point]]
    val_files = [clean_files[i] for i in indices[split_point:]]

    train_dataset = SpeechNoiseDataset(
        clean_audio_paths=train_files,
        noise_audio_paths=noise_files,
        snr_range=snr_range,
        is_validation=False
    )

    val_dataset = SpeechNoiseDataset(
        clean_audio_paths=val_files,
        noise_audio_paths=noise_files,
        snr_range=snr_range,
        is_validation=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
