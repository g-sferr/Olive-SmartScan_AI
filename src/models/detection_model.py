import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

# Modello in grado di predirre al massimo maxBBoxes
class OliveCNN(nn.Module):
    def __init__(self, maxBBoxes = 1):
        super(OliveCNN, self).__init__()
        self.maxBBoxes = maxBBoxes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # L'output della rete sarà composto da 4(coordinate) * maxBBoxesper
        self.bbox_fc = nn.Linear(10 * 10 * 512, self.maxBBoxes * 4)  # maxBBoxes * 4 coordinate per bounding box: x, y, w, h
        self.conf_fc = nn.Linear(10 * 10 * 512, self.maxBBoxes)  # maxBBoxes confidenze per bounding box
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 648 -> 324
        x = self.pool(self.relu(self.conv2(x)))  # 324 -> 162
        x = self.pool(self.relu(self.conv3(x)))  # 162 -> 81
        x = self.pool(self.relu(self.conv4(x)))  # 81 -> 40
        x = self.pool(self.relu(self.conv5(x)))  # 40 -> 20
        x = self.pool(self.relu(self.conv6(x)))  # 20 -> 10
        x = x.view(-1, 10 * 10 * 512)  # Flatten
        
        bbox = self.bbox_fc(x).view(-1, self.maxBBoxes, 4)
        conf = torch.sigmoid(self.conf_fc(x)).view(-1, self.maxBBoxes)  # La confidenza è una probabilità, quindi usiamo sigmoid
        
        return bbox, conf

    def filter_bboxes(self, outputBatchBBoxes, outputBatchConfs, conf_threshold = 0.5): # (16 * 5BB, 16*5Conf)
        # Filtra le bounding box in base alla confidenza
        filtered_bboxes = []
        filtered_confs = []
        for BBoxesOneImage, ConfsOneImage in zip(outputBatchBBoxes, outputBatchConfs):
            temp_filtered_bboxes = []
            temp_filtered_confs = []

            for singleBBox, signleConf in zip(BBoxesOneImage, ConfsOneImage):
                if signleConf > conf_threshold:
                    temp_filtered_bboxes.append(singleBBox[:4].tolist())
                    temp_filtered_confs.append(signleConf.item())
            filtered_bboxes.append(temp_filtered_bboxes)
            filtered_confs.append(temp_filtered_confs)
            
        return  torch.tensor(filtered_bboxes, dtype=torch.float32),  torch.tensor(filtered_confs, dtype=torch.float32)


'''
# MODELLO DA RIGUARDARE -> L'output qui è solamente una BoundingBox per immagine. 10/07/2024
# Il modello corretto deve dare in uscita un numero MASSIMO di boundingBoxes, che noi imponiamo essere uguale alla media belle boundingBoxes di tutto il dataset
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
'''

