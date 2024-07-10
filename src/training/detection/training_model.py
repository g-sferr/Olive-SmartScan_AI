import sys
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from src.models.detection_model import OliveCNN  # Classe che definisce il modello della rete cnn
from src.data_management.data_acquisition import OliveDatasetLoader 
from torch.utils.data import DataLoader

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
def training_steps(model, dataloader, criterion_bbox, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, bboxes in dataloader:
            # Trasferisci i tensori al dispositivo (CPU o GPU)
            inputs = inputs.to(device)
            bboxes = [bbox.to(device) for bbox in bboxes]

            outputs_bboxes = model(inputs)
            #loss = criterion_bbox(outputs_bboxes, bboxes)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            #running_loss += loss.item()

            loss_bbox = 0
            for output, target in zip(outputs_bboxes, bboxes):
                count = 0
                print(f"output.size(): {output.size()} | target.size(): {target.size()}")
                
                #Parte di codice da ridefinire meglio quando si sceglie l'architettura del modello
                
                if output.size() == target.size():
                    loss = criterion_bbox(output, target)
                    count += 1
                #if count > 0:
                #    loss /= count
                #loss = criterion_bbox(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #lossValue = loss.double()
                    loss_bbox += loss.item()#lossValue  # Ottieni il valore numerico della perdita
                    print(f"loss_bbox: {loss_bbox}")

            running_loss += loss_bbox * inputs.size(0)

            '''
            loss_complete = 0
            for output, target in zip(outputs_bboxes, bboxes):
                if output.size() == target.size():
                    loss += criterion_bbox(output, target)

            tot = len(outputs_bboxes)
            if tot != 0:
                loss_bbox = loss / tot
            else:
                loss_bbox = loss

           
            optimizer.step()
            
            running_loss += loss_bbox * inputs.size(0)
            '''
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    

def start_train():
    # ---------- DATA ACQUISITION & DATA PRE-PROCESSING ----------
    data_dir = 'datasets/processed/train_set/'
    datasetLoader = OliveDatasetLoader(data_dir)
    dataloader = DataLoader(datasetLoader, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # ---------- DATA PROCESSING ----------
    # Inizializzazione del modello
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OliveCNN()
    model = model.to(device)

    # Definizione della loss function e dell'ottimizzatore
    criterion_bbox = torchvision.ops.box_iou #nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    training_steps(model, dataloader, criterion_bbox, optimizer, device, num_epochs=10)

    torch.save(model.state_dict(), '../../../final_models/checkpoints/best_detection_model.pth')

def module_tester():
    # Codice per testare le funzioni del modulo
    start_train()

if __name__ == '__main__':
    module_tester()