import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torch.optim import lr_scheduler
from src.models.detection_model import OliveCNN  # Classe che definisce il modello della rete cnn
from src.models.alternativeModel import AlternativeModel
from src.data_management.data_acquisition import OliveDatasetLoader
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_fn(batch):
    images, bboxes = zip(*batch)
    # Stack images (images are already tensors of the same shape)
    images = torch.stack(images, dim=0)
    return images, list(bboxes)


# Funzione di addestramento
def training_steps(model, dataloader, testloader,
                   loss_criterion, optimizer, device, num_epochs, accumulate=1):
    # scaler = GradScaler()
    last_opt_step = 0
    ni = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm per il ciclo di avanzamento
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")

        for batch_idx, (inputs, targetBBoxes) in progress_bar:
            # Trasferisci i tensori al dispositivo (CPU o GPU)
            inputs = inputs.to(device)
            targetBBoxes = [bboxes.to(device) for bboxes in targetBBoxes]

            # Avvio del contesto autocast per precisione mista
            with autocast():
                # Forward pass
                outputBatchBBoxes, outputBatchConfs = model(inputs)

                # Filtro basato sulla confidenza
                # confidence = 0.2
                # filteredBatchBBoxes, filteredBatchConfs = model.filter_bboxes(outputBatchBBoxes, outputBatchConfs, confidence)
                # filteredBatchBBoxes = [fbboxes.to(device) for fbboxes in filteredBatchBBoxes]

                batch_loss = 0.0
                for i in range(len(inputs)):

                    targetBoxesOneImage = targetBBoxes[i]
                    outputBatchBBoxesOneImage = outputBatchBBoxes[i]

                    sizeTargetBBoxes = targetBoxesOneImage.size(0)
                    sizeOutputBBoxes = outputBatchBBoxesOneImage.size(0)

                    if sizeTargetBBoxes > sizeOutputBBoxes:
                        # Pad filteredBBoxesOneImage to match targetBoxesOneImage size
                        paddedOutputBBoxesOneImage = torch.zeros(sizeTargetBBoxes, 4, device=device)
                        paddedOutputBBoxesOneImage[:sizeOutputBBoxes, :4] = outputBatchBBoxesOneImage
                        outputBatchBBoxesOneImage = paddedOutputBBoxesOneImage.detach().requires_grad_(True)
                    elif sizeOutputBBoxes > sizeTargetBBoxes:
                        # Pad targetBoxesOneImage to match filteredBBoxesOneImage size
                        paddedTargetBoxesOneImage = torch.zeros(sizeOutputBBoxes, 4, device=device)
                        paddedTargetBoxesOneImage[:sizeTargetBBoxes, :4] = targetBoxesOneImage
                        targetBoxesOneImage = paddedTargetBoxesOneImage.detach().requires_grad_(True)

                    # Calcolo della perdita
                    committedError = loss_criterion(outputBatchBBoxesOneImage, targetBoxesOneImage)
                    batch_loss += committedError

            # Scaling dei gradienti e backward pass
            batch_loss.backward()

            ni += 1  # Incrementa il contatore globale dei passi
            if ((ni - last_opt_step) >= accumulate) or (
                    (batch_idx + 1) == len(dataloader)):  # Check per aggiornamento dell'ottimizzatore
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Clip dei gradienti
                optimizer.step()
                optimizer.zero_grad()

                last_opt_step = ni

            running_loss += batch_loss.item()

            # Aggiornamento della descrizione della barra di progresso con la loss corrente
            progress_bar.set_postfix(batch_loss=batch_loss.item())

        epoch_loss = running_loss / len(dataloader)

        # Calcolo dell'accuratezza
        if testloader is not None:
            acc = 0  # accumulerà la somma delle previsioni corrette
            count = 0  # tiene traccia del numero totale di campioni
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = [bboxes.to(device) for bboxes in labels]  # labels.to(device)
                output_pred, output_confs = model(inputs)
                acc += (torch.argmax(output_pred,
                                     1) == labels).float().sum().item()  # trova l'indice della previsione più alta per ogni campione
                count += len(labels)
            acc /= count  # calcolo dell'accuratezza media
            print(f'Epoch {epoch + 1}: model accuracy {acc * 100:.2f}%')

        print("")
        print(f'Total Loss for Epoch {epoch + 1}/{num_epochs}:  {epoch_loss:.4f}')
        print(
            "------------------------------------------------------------------------------------------------------------------------")


def start_train():
    # ---------- DATA ACQUISITION & DATA PRE-PROCESSING ----------
    print("")
    print('[ Step (1): ********** Path Scanning for "train_set" data ********** ]')
    batch_size = 8
    data_dir = '../../../datasets/processed/train_set/'
    datasetLoader = OliveDatasetLoader(data_dir)
    dataloader = DataLoader(datasetLoader, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print("")
    print("[ Step (2): ********** DATASET LOADED ********** ]")
    print("")

    # data_dir = 'datasets/processed/test_set/'
    # datasetLoader = OliveDatasetLoader(data_dir)
    # testloader = DataLoader(datasetLoader, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # ---------- DATA PROCESSING ----------
    # Inizializzazione del modello
    print(f"( --- RESPONSE CHECK CUDA DEVICE AVAILABILITY:  {torch.cuda.is_available()} --- )")
    print("")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = OliveCNN(3) # Max 5 BoundingBoxes predette
    model = AlternativeModel(10)
    model = model.to(device)
    print("[ Step (3): ********** MODEL TRANSFERRED ON GPU DEVICE ********** ]")
    print("")

    # Definizione della loss function e dell'ottimizzatore
    criterion_bbox = nn.MSELoss()
    # Optimizer with batch tuning
    nominal_batch_size = 16  # nominal batch size:
    accumulate = max(round(nominal_batch_size / batch_size), 1)  # accumulate loss before optimizing
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_steps(model, dataloader, None,
                   criterion_bbox, optimizer, device,
                   num_epochs=100, accumulate=accumulate)

    model_out_dir = os.path.abspath('../../../final_models/checkpoints/detection_model.pth')
    torch.save(model.state_dict(), model_out_dir)

    print("")
    print("[ Final Step (4): ########## TRAINING PROCESS for 'Olive-CNN' FINISHED ########## ]")
    print("")
    print("[ Final Trained Model Saved in: 'src/final_models/checkpoints/detection_model.pth' ]")
    print("")


def module_tester():
    # Codice per testare le funzioni del modulo
    start_train()


if __name__ == '__main__':
    module_tester()
