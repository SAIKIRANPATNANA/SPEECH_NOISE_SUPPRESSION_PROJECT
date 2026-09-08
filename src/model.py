"""
model.py - 2D U-Net Architecture for Audio Noise Suppression
============================================================

A CSE Guide to the U-Net Architecture:
--------------------------------------
1. What is U-Net?
   U-Net is an Encoder-Decoder neural network with "Skip Connections" that connect
   each layer in the encoder directly to the matching layer in the decoder.
   
   Original paper: Ronneberger et al. (2015) for biomedical image segmentation.
   In audio AI: Used in SOTA systems (Spleeter, Demucs, Wave-U-Net).

2. Why do we need Skip Connections for Audio?
   - The Encoder repeatedly downsamples the 2D spectrogram (256x256 -> 8x8).
     This extracts high-level semantic context ("someone is speaking a vowel").
     BUT downsampling destroys fine spatial details!
   - If we simply upsample without skip connections (plain Autoencoder), the output
     spectrogram is blurry. Blurry spectrogram = muffled, underwater-sounding audio!
   - Skip Connections act like bypass wires:
     They copy crisp high-frequency details (consonants 's', 't', 'k', pitch harmonics)
     directly across from the encoder to the decoder via torch.cat().

3. Masking vs Direct Spectral Mapping:
   - Masking (Recommended):
     The network outputs an attenuation mask M[f, t] in [0.0, 1.0] using Sigmoid.
     Clean Estimate: S_hat = M * Y (element-wise multiplication with noisy input).
     Why it works better: It acts as an acoustic dimmer switch. The network doesn't
     have to synthesize speech from scratch; it only decides how much noise to mute!
   - Direct Mapping:
     The network directly outputs S_hat using ReLU. Harder to train because the
     dynamic range is unbounded.

4. Visual ASCII Diagram of U-Net Data Flow:
   
   Noisy Spectrogram [1, 256, 256]
          |
        [Enc1] -----------------(Skip 1)-----------------> [Dec1] -> Output Mask [1, 256, 256]
          v                                                  ^                     |
        [Enc2] --------------(Skip 2)--------------> [Dec2]  |                     v
          v                                            ^     |             Clean = Mask * Noisy
        [Enc3] ------------(Skip 3)----------> [Dec3]  |     |
          v                                      ^     |     |
        [Enc4] --------(Skip 4)--------> [Dec4]  |     |     |
          v                                ^     |     |     |
      [Bottleneck: 512, 8, 8] -------------+     |     |     |
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Encoder block: Conv2d (downsampling by factor of 2) + BatchNorm2d + LeakyReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_norm
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    Decoder block: ConvTranspose2d (upsampling by factor of 2) + BatchNorm2d + ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSpeechEnhancer(nn.Module):
    """
    Complete 2D Audio U-Net for Speech Noise Suppression.
    
    Accepts 1-channel magnitude spectrograms [Batch, 1, 256, 256].
    Outputs enhanced magnitude spectrogram [Batch, 1, 256, 256] and predicted mask.
    """

    def __init__(self, mode: str = 'masking'):
        """
        Parameters:
            mode (str): 'masking' (predicts IRM in [0, 1]) or 'mapping' (predicts clean magnitude directly)
        """
        super().__init__()
        if mode not in ('masking', 'mapping'):
            raise ValueError(f"Invalid mode: {mode}. Choose 'masking' or 'mapping'.")
        self.mode = mode

        # --- ENCODER (Contracting Path) ---
        # Input: [B, 1, 256, 256]
        self.enc1 = ConvBlock(1, 32, use_norm=False)   # -> [B, 32, 128, 128]
        self.enc2 = ConvBlock(32, 64, use_norm=True)   # -> [B, 64, 64, 64]
        self.enc3 = ConvBlock(64, 128, use_norm=True)  # -> [B, 128, 32, 32]
        self.enc4 = ConvBlock(128, 256, use_norm=True) # -> [B, 256, 16, 16]

        # --- BOTTLENECK (Deepest Latent Representation) ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), # -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )

        # --- DECODER (Expanding Path with Skip Connections) ---
        # Dec4 input: 512 -> output: 256. Concat with enc4 (256) => 512 channels
        self.dec4 = DeconvBlock(512, 256, use_dropout=True)
        # Dec3 input: 512 (256 dec4 + 256 enc4) -> output: 128. Concat with enc3 (128) => 256 channels
        self.dec3 = DeconvBlock(512, 128, use_dropout=True)
        # Dec2 input: 256 (128 dec3 + 128 enc3) -> output: 64. Concat with enc2 (64) => 128 channels
        self.dec2 = DeconvBlock(256, 64, use_dropout=False)
        # Dec1 input: 128 (64 dec2 + 64 enc2) -> output: 32. Concat with enc1 (32) => 64 channels
        self.dec1 = DeconvBlock(128, 32, use_dropout=False)

        # --- FINAL OUTPUT LAYER ---
        # Final conv upsamples [B, 64, 128, 128] -> [B, 1, 256, 256]
        self.final_conv = nn.ConvTranspose2d(
            64,
            1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True
        )

        if self.mode == 'masking':
            self.output_act = nn.Sigmoid()  # IRM mask values in range [0.0, 1.0]
        else:
            self.output_act = nn.ReLU()     # Direct magnitude estimation >= 0.0

    def forward(
        self,
        noisy_mag: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Parameters:
            noisy_mag: Tensor of shape [Batch, 1, 256, 256]

        Returns:
            enhanced_mag: Tensor of shape [Batch, 1, 256, 256]
            predicted_mask: Tensor of shape [Batch, 1, 256, 256] (if masking mode) or None
        """
        # Encoder forward pass with skip activations saved
        e1 = self.enc1(noisy_mag)  # [B, 32, 128, 128]
        e2 = self.enc2(e1)         # [B, 64, 64, 64]
        e3 = self.enc3(e2)         # [B, 128, 32, 32]
        e4 = self.enc4(e3)         # [B, 256, 16, 16]

        # Bottleneck
        b = self.bottleneck(e4)    # [B, 512, 8, 8]

        # Decoder forward pass with skip connection concatenation
        d4 = self.dec4(b)          # [B, 256, 16, 16]
        d4 = torch.cat([d4, e4], dim=1) # [B, 512, 16, 16]

        d3 = self.dec3(d4)         # [B, 128, 32, 32]
        d3 = torch.cat([d3, e3], dim=1) # [B, 256, 32, 32]

        d2 = self.dec2(d3)         # [B, 64, 64, 64]
        d2 = torch.cat([d2, e2], dim=1) # [B, 128, 64, 64]

        d1 = self.dec1(d2)         # [B, 32, 128, 128]
        d1 = torch.cat([d1, e1], dim=1) # [B, 64, 128, 128]

        # Final prediction
        raw_output = self.final_conv(d1) # [B, 1, 256, 256]
        output = self.output_act(raw_output)

        if self.mode == 'masking':
            mask = output
            # Clean speech estimate = Mask * Noisy Speech
            enhanced_mag = mask * noisy_mag
            return enhanced_mag, mask
        else:
            enhanced_mag = output
            return enhanced_mag, None

    def get_num_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
