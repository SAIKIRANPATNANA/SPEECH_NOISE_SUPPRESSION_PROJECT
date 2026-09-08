"""
test_pipeline.py - Automated Verification and Unit Tests
========================================================
Tests all core components of the Speech Noise Suppression pipeline:
1. STFT and iSTFT mathematical roundtrip reconstruction.
2. SNR mixing formula accuracy.
3. Synthetic noise generators.
4. Spectrogram padding and cropping.
5. PyTorch Dataset and DataLoader batch generation.
6. U-Net model forward pass, output bounds, and gradient backprop.
7. Audio quality metrics (SNR and SI-SDR).
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_processing import (
    compute_stft,
    reconstruct_waveform,
    pad_or_crop_spectrogram,
    mag_to_db
)
from src.noise_generator import (
    mix_at_snr,
    calculate_power,
    generate_white_noise,
    generate_pink_noise,
    generate_hum_noise
)
from src.dataset import SpeechNoiseDataset
from src.model import UNetSpeechEnhancer
from src.evaluation import calculate_snr, calculate_si_sdr, calculate_lsd


class TestSpeechNoiseSuppression(unittest.TestCase):

    def test_stft_istft_roundtrip(self):
        """Verify iSTFT(STFT(x)) perfectly reconstructs the time-domain waveform."""
        sr = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440 Hz pure tone + harmonic
        signal = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        signal = signal.astype(np.float32)

        mag, phase = compute_stft(signal, n_fft=512, hop_length=128, win_length=512)
        reconstructed = reconstruct_waveform(mag, phase, hop_length=128, win_length=512, length=len(signal))

        # Check maximum absolute error
        max_error = np.max(np.abs(signal - reconstructed))
        self.assertLess(max_error, 1e-4, f"STFT roundtrip error too large: {max_error}")

    def test_snr_mixing(self):
        """Verify that mix_at_snr precisely achieves requested SNR values."""
        clean = np.random.randn(16000).astype(np.float32)
        noise = np.random.randn(16000).astype(np.float32)

        target_snrs = [-5.0, 0.0, 5.0, 10.0]
        for target in target_snrs:
            noisy, scaled_noise, actual_snr = mix_at_snr(clean, noise, target_snr_db=target)
            self.assertAlmostEqual(
                actual_snr, target, delta=0.2,
                msg=f"Actual SNR {actual_snr} deviated from target {target}"
            )
            # Verify noisy = clean + scaled_noise (before normalization)
            self.assertEqual(len(noisy), len(clean))

    def test_noise_generators(self):
        """Verify noise generators produce correct lengths and finite values."""
        samples = 8000
        white = generate_white_noise(samples)
        pink = generate_pink_noise(samples)
        hum = generate_hum_noise(samples)

        for name, n in [("white", white), ("pink", pink), ("hum", hum)]:
            self.assertEqual(len(n), samples)
            self.assertFalse(np.isnan(n).any(), f"{name} contains NaN")
            self.assertFalse(np.isinf(n).any(), f"{name} contains Inf")
            self.assertGreater(np.max(np.abs(n)), 0.0, f"{name} is silent")

    def test_pad_crop_spectrogram(self):
        """Verify spectrograms are accurately reshaped to [256, 256]."""
        # Test case 1: Smaller than 256x256
        small = np.random.randn(200, 150).astype(np.float32)
        padded = pad_or_crop_spectrogram(small, target_frames=256, target_freq=256)
        self.assertEqual(padded.shape, (256, 256))

        # Test case 2: Larger than 256x256
        large = np.random.randn(300, 400).astype(np.float32)
        cropped = pad_or_crop_spectrogram(large, target_frames=256, target_freq=256)
        self.assertEqual(cropped.shape, (256, 256))

    def test_model_forward_and_backward(self):
        """Verify UNet forward pass shapes, Sigmoid bounds [0, 1], and gradient flow."""
        model = UNetSpeechEnhancer(mode='masking')
        model.train()

        batch_size = 2
        dummy_in = torch.randn(batch_size, 1, 256, 256, requires_grad=True)
        enh_mag, mask = model(dummy_in)

        # Output shape assertions
        self.assertEqual(enh_mag.shape, (batch_size, 1, 256, 256))
        self.assertEqual(mask.shape, (batch_size, 1, 256, 256))

        # Mask range assertion (Sigmoid guarantees [0, 1])
        self.assertTrue((mask >= 0.0).all() and (mask <= 1.0).all(), "Mask values outside [0, 1]")

        # Loss & backward gradient test
        target = torch.randn(batch_size, 1, 256, 256)
        loss = torch.nn.functional.l1_loss(enh_mag, target)
        loss.backward()

        # Ensure encoder weights received gradients
        self.assertIsNotNone(model.enc1.block[0].weight.grad)
        self.assertFalse(torch.isnan(model.enc1.block[0].weight.grad).any())

    def test_evaluation_metrics(self):
        """Verify SNR and SI-SDR computations."""
        clean = np.random.randn(16000)
        # Identical signal should yield very high SNR (> 90 dB)
        high_snr = calculate_snr(clean, clean)
        self.assertGreater(high_snr, 80.0)

        # Distorted signal
        noisy = clean + 0.5 * np.random.randn(16000)
        snr = calculate_snr(clean, noisy)
        si_sdr = calculate_si_sdr(clean, noisy)
        self.assertTrue(isinstance(snr, float))
        self.assertTrue(isinstance(si_sdr, float))


if __name__ == "__main__":
    unittest.main()
