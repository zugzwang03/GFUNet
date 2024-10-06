import torch
import torch.nn as nn
import torch.nn.functional as F

lowOrHigh = 'low'

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64


        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def frequency_transform(self, x):
        # Apply FFT and return the magnitude of the frequency components
        freq = torch.fft.fft2(x)
        magnitude = torch.abs(freq)  # Convert to real values
        return magnitude

    def frequency_filter(self, frequency_maps, type='low', threshold=0.01):
        if type == 'low':
            # Apply low-pass filter (keep low frequencies)
            return frequency_maps * (abs(frequency_maps) < threshold)  # custom threshold
        elif type == 'high':
            # Apply high-pass filter (keep high frequencies)
            return frequency_maps * (abs(frequency_maps) >= threshold)

    def frequency_to_spatial(self, frequency_maps):
        # Apply Inverse 2D FFT to go back to spatial domain
        spatial_maps = torch.fft.ifft2(frequency_maps)
        spatial_maps = torch.abs(spatial_maps)
        return spatial_maps

    def fuse_features(self, spatial_features, frequency_features, channel_size):

        conv = nn.Conv2d(channel_size, channel_size // 2, kernel_size=3, padding=1)
        spatial_features = conv(spatial_features)
        frequency_features = conv(frequency_features)

        # Concatenate the spatial and frequency domain features along the channel dimension
        fused_features = torch.cat([spatial_features, frequency_features], dim=1)

        return fused_features

    def fusion(self, feature_map, channel_size):
        frequency_feature = self.frequency_transform(feature_map)
        frequency_feature = self.frequency_filter(frequency_feature, lowOrHigh, 0.01)
        spatial_feature = self.frequency_to_spatial(frequency_feature)
        return self.fuse_features(spatial_feature, frequency_feature, channel_size)
        return feature_map

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        # xp1 = self.fusion(xp1, 64)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        # xp2 = self.fusion(xp2, 128)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        # xp3 = self.fusion(xp3, 256)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)
        # xp4 = self.fusion(xp4, 512)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))
        xe52 = self.fusion(xe52, 1024)

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))
        # xd12 = self.fusion(xd12, 512)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))
        # xd22 = self.fusion(xd22, 256)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))
        # xd32 = self.fusion(xd32, 128)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))
        xd42 = self.fusion(xd42, 64)

        # Output layer
        out = self.outconv(xd42)

        return out

# Initialize the model
model = UNet(n_class=1)

# Print the model architecture
print(model)
