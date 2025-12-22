import torch
import torch.nn as nn
import torch.nn.functional as F

class ClimateCNN(nn.Module):
    def __init__(self, height, width):
        super(ClimateCNN, self).__init__()
        self.height = height
        self.width = width
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.enc1(x))
        x = self.pool(x)
        x = self.relu(self.enc2(x))
        x = self.pool(x)
        
        # Decoder
        x = self.dec1(x)
        
        # --- CORRECCIÓN CLAVE ---
        # Forzamos la interpolación al tamaño EXACTO original (721x1440)
        # Esto evita el error de perder 1 pixel por ser impar
        x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=False)
        
        x = self.dec2(x)
        return x

class ClimateMLP(nn.Module):
    def __init__(self, height, width):
        super(ClimateMLP, self).__init__()
        # Conv2d 1x1 actúa como un MLP pixel a pixel
        self.layer1 = nn.Conv2d(1, 128, kernel_size=1) 
        self.layer2 = nn.Conv2d(128, 64, kernel_size=1)
        self.layer3 = nn.Conv2d(64, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
