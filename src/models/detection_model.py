import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

def test():
    print("hello")

class OliveCNN(nn.Module):
    def __init__(self):
        super(OliveCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # L'output per le coordinate delle bounding boxes e le probabilità
        self.bbox_fc = nn.Linear(10 * 10 * 512, 4)  # 4 coordinate per ogni bounding box: x, y, w, h
        self.conf_fc = nn.Linear(10 * 10 * 512, 1)  # 1 confidenza per ogni bounding box
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 648 -> 324
        x = self.pool(self.relu(self.conv2(x))) # 324 -> 162
        x = self.pool(self.relu(self.conv3(x))) # 162 -> 81
        x = self.pool(self.relu(self.conv4(x))) # 81 -> 40
        x = self.pool(self.relu(self.conv5(x))) # 40 -> 20
        x = self.pool(self.relu(self.conv6(x))) # 20 -> 10
        x = x.view(-1, 10 * 10 * 512) # Flatten
        
        bbox = self.bbox_fc(x)
        conf = torch.sigmoid(self.conf_fc(x))  # La confidenza è una probabilità, quindi usiamo sigmoid
        
        return bbox, conf


