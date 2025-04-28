import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super.__init__()
        chan0 = in_channels
        chan1 = 2 * in_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=chan0, out_channels=chan1, kernel_size=3, stride=1, padding=1)
            nn.BatchNorm2d(chan1)
            nn.ReLU()
            nn.Conv2d(in_channels=chan1, out_channels=chan1, kernel_size=3, stride=1, padding=1)
            nn.BatchNorm2d(chan1)
        )
        self.skip = nn.Conv2d(in_channels=chan0, out_channels=chan1, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = self.skip(x) + self.block(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.skip3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.skip4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)

        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.skip5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)

        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.avepool = nn.AvgPool2d(kernel_size=7, stride=2, padding=0)

        self.fc1 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        skip = x
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = x + skip
        x = self.relu(x)

        skip3 = self.skip3(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = x + skip3
        x = self.relu(x)

        skip4 = self.skip4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = x + skip4
        x = self.relu(x)

        skip5 = self.skip5(x)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = x + skip5
        x = self.relu(x)

        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)

        return x
