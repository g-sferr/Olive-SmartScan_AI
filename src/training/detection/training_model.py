import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_management.data_acquisition import split_data
from data_management.data_preprocessing import load_dataset
from models.detection_model import OliveCNN  # Assumendo che tu abbia già il modello definito in model.py

def training_phase(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Ogni epoca ha una fase di training e una di validazione
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

def start_train():
    # Split dei dati
    split_data('data/raw')

    # Caricamento dei dati
    dataloaders = {
        'train': load_dataset('data/processed/train'),
        'val': load_dataset('data/processed/val', shuffle=False)
    }

    # Inizializzazione del modello
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OliveCNN()
    model = model.to(device)

    # Definizione della loss function e dell'ottimizzatore
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training del modello
    model = training_phase(model, dataloaders, criterion, optimizer, num_epochs=25)

    # Salvataggio del modello
    torch.save(model.state_dict(), 'final_models/checkpoints/best_model.pth')



def module_tester():
    #Code for test functions of the module
    start_train()


    
if __name__ == '__main__':
    module_tester()