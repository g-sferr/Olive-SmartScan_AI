import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_management.data_acquisition import split_data
from data_management.data_preprocessing import load_dataset
from models.detection_model import OliveCountRegressor  # Assumendo che tu abbia già il modello definito in model.py



def training_phase(model, dataloaders, criterion, optimizer, num_epochs=25):
    return

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
    model = OliveCountRegressor()
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