import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 1, bias=True)

    def forward(self, input):
        x = self.model(input)
        return x.view(-1)

class AlternativeModel(nn.Module):
    def __init__(self, num_anchors=9):
        super(AlternativeModel, self).__init__()
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)
 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
 
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flat = nn.Flatten()
 
        # Fully connected layers for bounding box regression
        self.fc_bbox = nn.Linear(512 * 10 * 10, num_anchors * 4)  # 4 coordinate per ogni bounding box: x, y, w, h
        self.fc_conf = nn.Linear(512 * 10 * 10, num_anchors)
 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))  # 640 -> 320
        x = self.drop1(x)
        x = self.pool2(self.act2(self.conv2(x)))  # 320 -> 160
        x = self.pool3(self.act3(self.conv3(x)))  # 160 -> 80
        x = self.pool4(self.act4(self.conv4(x)))  # 80 -> 40
        x = self.pool5(self.act5(self.conv5(x)))  # 40 -> 20
        x = self.pool6(self.act6(self.conv6(x)))  # 20 -> 10
        x = self.flat(x)
        bbox = self.fc_bbox(x)
        bbox = bbox.view(-1, self.num_anchors, 4)  # Reshape to (batch_size, num_anchors, 4)
 
        conf = self.fc_conf(x)
        conf = self.sigmoid(conf)  # Sigmoid for confidence to get probabilities
        conf = conf.view(-1, self.num_anchors)  # Reshape to (batch_size, num_anchors)
 
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
        