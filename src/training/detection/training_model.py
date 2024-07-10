import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from src.models.detection_model import OliveCNN  # Classe che definisce il modello della rete cnn
from src.data_management.data_acquisition import OliveDatasetLoader 
from torch.utils.data import DataLoader

# Funzione di addestramento
def training_steps(model, dataloader, criterion_bbox, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, bboxes in dataloader:

            # Trasferisci i tensori al dispositivo (CPU o GPU)
            inputs = inputs.to(device)
            bboxes = bboxes.to(device)
            #confs = confs.to(device)

            optimizer.zero_grad()
            outputs_bbox = model(inputs)
            loss_bbox = criterion_bbox(outputs_bbox, bboxes)
            # loss_conf = criterion_conf(outputs_conf, confs)
            loss = loss_bbox # + loss_conf
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def start_train():
    # ---------- DATA ACQUISITION & DATA PRE-PROCESSING ----------
    data_dir = '../../../datasets/processed/train_set/'
    datasetLoader = OliveDatasetLoader(data_dir)
    dataloader = DataLoader(datasetLoader, batch_size=32, shuffle=True)
    
    # ---------- DATA PROCESSING ----------
    # Inizializzazione del modello
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OliveCNN()
    model = model.to(device)

    # Definizione della loss function e dell'ottimizzatore
    criterion_bbox = nn.MSELoss()
    # criterion_conf = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    training_steps(model, dataloader, criterion_bbox, optimizer, device, num_epochs=25)

    torch.save(model.state_dict(), '../../../final_models/checkpoints/best_detection_model.pth')


def module_tester():
    #Code for test functions of the module
    start_train()


if __name__ == '__main__':
    module_tester()