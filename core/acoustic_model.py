import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class VoiceShieldAcousticCNN(nn.Module):
    """
    Multi-Scale Deepfake & Synthesis Artifact Classifier.
    Analyzes input Mel-spectrogram tensors of shape (Batch, 1, N_MELS, TARGET_FRAMES).
    """
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=(1, 2), padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = ConvBlock(32, 48, stride=2)   # downsample
        self.layer2 = ConvBlock(48, 64, stride=2)   # downsample
        self.layer3 = ConvBlock(64, 128, stride=2)  # downsample
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 1, 80, 96)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        logits = self.head(x)
        return logits

    def predict_probability(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
        return prob.squeeze(1).cpu().numpy()
