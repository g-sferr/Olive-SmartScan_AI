import sys
import os
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from src.models.detection_model import OliveCNN  # Classe che definisce il modello della rete cnn
from src.models.alternativeModel import AlternativeModel
from src.data_management.data_acquisition import OliveDatasetLoader 
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

class CustomMSELoss(nn.Module): # Classe Da Eliminare 10/07/2024
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        loss = 0
        for output, target in zip(outputs, targets):
            if output.size() == target.size():
                loss += self.mse_loss(output, target)

        tot = len(outputs)
        if tot != 0:
            return loss / len(outputs)
        return loss
    
    def backward(self):
        self.backward()

def collate_fn(batch):
    images, bboxes = zip(*batch)
    
    # Stack images (images are already tensors of the same shape)
    images = torch.stack(images, dim=0)
    
    # Create list of bounding boxes
    #bboxes = [bbox for bbox in bboxes]

    return images, list(bboxes)


# Funzione di addestramento
def training_steps(model, dataloader, testloader, loss_criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targetBBoxes in dataloader:
            # Trasferisci i tensori al dispositivo (CPU o GPU)
            inputs = inputs.to(device)
            targetBBoxes = [bboxes.to(device) for bboxes in targetBBoxes]

            outputBatchBBoxes, outputBatchConfs = model(inputs)
            
            confidence = 0
            filteredBatchBBoxes, filteredBatchConfs = model.filter_bboxes(outputBatchBBoxes, outputBatchConfs, confidence)
            filteredBatchBBoxes = filteredBatchBBoxes.to(device)
            #print(f"filteredBatchBBoxes.size(): {filteredBatchBBoxes.size()}")
            ''' STAMPA
            for BBoxesOneImage, ConfsOneImage in zip(filteredBatchBBoxes, filteredBatchConfs):
                print(f"Foto: {count} | output: {BBoxesOneImage.size()}")
                for singleBBox, confBBox in zip(BBoxesOneImage, ConfsOneImage):
                    x, y, w, h = singleBBox
                    print(f"x={x}, y={y}, w={w}, h={h} | Conf: {confBBox}")
                count+=1
            '''
            '''
            for targetBoxesOneImage in targetBBoxes :
                #targetBoxesOneImageAsTensor = torch.tensor(targetBoxesOneImage, dtype=torch.float32) # --> tensor(NBoundingBoxes, 4) : [ [x1, y1, w1, h1], [x2, y2, w2, h2] ... ]
                for filteredBBoxesOneImage in filteredBatchBBoxes:
                    targetBoxesOneImageAsTensorCPY = targetBoxesOneImage.clone().detach().requires_grad_(True)
                    filteredBBoxesOneImageCPY = filteredBBoxesOneImage.clone().detach().requires_grad_(True)
                    sizeTargetBBoxes = targetBoxesOneImageAsTensorCPY.size(0)
                    sizeFilteredBBoxes = filteredBBoxesOneImage.size(0)

                    if sizeTargetBBoxes > sizeFilteredBBoxes:
                        #print("sizeTargetBBoxes > sizeFilteredBBoxes:")
                        filteredBBoxesOneImageCPY = torch.zeros(sizeTargetBBoxes, 4, dtype=torch.float32)
                        filteredBBoxesOneImageCPY[:sizeFilteredBBoxes, :filteredBBoxesOneImage.size(1)] = filteredBBoxesOneImage # Dopo, i due tensor hanno stessa size
                        #print(f"filteredBBoxesOneImage.size: {filteredBBoxesOneImage.size()} | targetBoxesOneImage.size(): {targetBoxesOneImage.size()}")
                    elif sizeFilteredBBoxes > sizeTargetBBoxes :
                        #print("sizeFilteredBBoxes > sizeTargetBBoxes :")
                        targetBoxesOneImageAsTensorCPY = torch.zeros(sizeFilteredBBoxes, 4, dtype=torch.float32)
                        targetBoxesOneImageAsTensorCPY[:sizeTargetBBoxes, :targetBoxesOneImage.size(1)] = targetBoxesOneImage # Dopo, i due tensor hanno stessa size
                        #print(f"filteredBBoxesOneImage.size: {filteredBBoxesOneImage.size()} | targetBoxesOneImageAsTensor.size(): {targetBoxesOneImage.size()}")

                    #print(f"CONTROPROVA: filtSizeCPY={filteredBBoxesOneImageCPY.size()} | targetSizeCPY: {targetBoxesOneImageAsTensorCPY.size()}")
                    
                    committedError = loss_criterion(filteredBBoxesOneImageCPY, targetBoxesOneImageAsTensorCPY)                    
                    optimizer.zero_grad()
                    committedError.backward()
                    optimizer.step()

                    running_loss += committedError.item() * inputs.size(0)
            '''
            loss = 0.0
            for filteredBBoxesOneImage in filteredBatchBBoxes:
                for targetBoxesOneImage in targetBBoxes :
                    committedErrorTensor = loss_criterion(filteredBBoxesOneImage, targetBoxesOneImage)
                    optimizer.zero_grad()
                    loss += committedErrorTensor.item()
                    optimizer.step()

            loss.backward()
            '''
            for targetBoxesOneImage in targetBBoxes :
                for filteredBBoxesOneImage in filteredBatchBBoxes:
                    committedErrorTensor = loss_criterion(filteredBBoxesOneImage.detach().requires_grad_(True), targetBoxesOneImage.detach().requires_grad_(True))            
                    
                    for committedError in committedErrorTensor:
                        for signleCommittedError in committedError:
                            signleCommittedError.detach().requires_grad_(True).backward()
                            
                    

            '''
            '''
                     for singleTargetBB in targetBoxesOneImage:
                         for singleFilteredBB in filteredBBoxesOneImage:
                            singleTargetBB = singleTargetBB.detach().requires_grad_(True).unsqueeze(0)
                            singleFilteredBB = singleFilteredBB.detach().requires_grad_(True).unsqueeze(0)
                            #committedError = loss_criterion(singleTargetBB.detach().requires_grad_(True), singleFilteredBB.detach().requires_grad_(True))            
                            committedError = loss_criterion(singleTargetBB, singleFilteredBB)            
                            optimizer.zero_grad()
                            committedError.backward()
                            optimizer.step()

                            running_loss += committedError.item() * inputs.size(0)
                    '''


                    
        '''
        acc = 0
        count = 0
        for inputNeverSeen, labels in testloader:
            print(f"")
            #outputBatchBBoxes, outputBatchConfs = model(inputs)
            #acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            #count += len(labels)
        acc /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100)) 
        '''
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    



def start_train():
    # ---------- DATA ACQUISITION & DATA PRE-PROCESSING ----------
    data_dir = 'datasets/processed/train_set/'
    datasetLoader = OliveDatasetLoader(data_dir)
    dataloader = DataLoader(datasetLoader, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    #data_dir = 'datasets/processed/test_set/'
    #datasetLoader = OliveDatasetLoader(data_dir)
    #testloader = DataLoader(datasetLoader, batch_size=16, shuffle=True, collate_fn=collate_fn)
    # ---------- DATA PROCESSING ----------
    # Inizializzazione del modello
    print(f"CUDA AVAILABLE ? -> {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = OliveCNN(3) # Max 5 BoundingBoxes predette
    model = AlternativeModel(3)
    model = model.to(device)
    criterion_bbox = box_iou #nn.MSELoss()
    # Definizione della loss function e dell'ottimizzatore
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    training_steps(model, dataloader, None , criterion_bbox, optimizer, device, num_epochs = 30)

    model_out_dir = os.path.abspath('final_models/checkpoints/best_detection_model.pth')
    torch.save(model.state_dict(), model_out_dir)

def module_tester():
    # Codice per testare le funzioni del modulo
    start_train()

if __name__ == '__main__':
    module_tester()