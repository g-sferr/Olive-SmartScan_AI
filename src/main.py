
import cv2
import os
import torch
import numpy as np
import torch.nn as nn
from math import sqrt
from statistics import stdev
from ultralytics import YOLO
import torchvision.models as models
from sklearn.metrics import mean_squared_error
from src.data_management.data_acquisition import OliveDatasetLoader


def compute_mse_and_devStd(true_counts, predicted_counts):
    
    mse = mean_squared_error(true_counts, predicted_counts)
    # std_dev = stdev(true_counts - predicted_counts)
    rmsd = sqrt(mse)

    return mse, rmsd

def is_contained(box1, box2):
    """Verifica se una bounding box è completamente contenuta in un'altra.

    Args:
        box1 (list): Coordinate della prima bounding box (x1, y1, x2, y2).
        box2 (list): Coordinate della seconda bounding box (x1, y1, x2, y2).

    Returns:
        bool: True se box1 è completamente contenuta in box2, False altrimenti.
    """
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and
            box1[2] <= box2[2] and box1[3] <= box2[3])

def count_olives(img_path, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = model(img_path)
    
    boxes = results[0].boxes
    boxes = boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()

    # Classi
    TREE_CLASS_ID = 0
    CROWN_CLASS_ID = 1
    OLIVE_CLASS_ID = 2

    # Separazione bounding box in base alle classi
    tree_boxes = [box for i, box in enumerate(boxes) if cls[i] == TREE_CLASS_ID]
    crown_boxes = [box for i, box in enumerate(boxes) if cls[i] == CROWN_CLASS_ID]
    olive_boxes = [box for i, box in enumerate(boxes) if cls[i] == OLIVE_CLASS_ID]

    olive_count = 0
    for olive_box in olive_boxes:
        for crown_box in crown_boxes:
            if is_contained(olive_box, crown_box):
                if tree_boxes:
                    # Verifica se la chioma è contenuta nell'albero
                    for tree_box in tree_boxes:
                        if is_contained(crown_box, tree_box):
                            olive_count += 1
                            break  # Esci dal loop dopo aver trovato un albero contenente la chioma
                else:
                    # Se non ci sono alberi, conta comunque l'oliva
                    olive_count += 1

    return olive_count

def main():
    #Code for test functions of the module
    data_dir = r'C:\path\images'
    oliveDatasetLoader = OliveDatasetLoader(data_dir)
    subFolder = 'FotoDiProvaModelloX'
    dataloader = oliveDatasetLoader._load_data(subFolder)

    # Carica il modello YOLO pre-addestrato
    model = YOLO(r'C:\path\ModelloX\best.pt')
    
    true_counts = []
    predicted_counts = []
    # Processa ogni immagine
    for images in dataloader:
        imagePath = os.path.join(data_dir, subFolder)
        imagePath = os.path.join(imagePath, images) 

        # Aggiunge il conteggio stimato
        count = count_olives(imagePath, model)
        print(f"Numero di olive PREDETTE per {images}: {count} ")
        predicted_counts.append(count)

        # nome del file txt corrispondente
        txt_file = os.path.splitext(imagePath)[0] + ".txt"

        # Lettura conteggio reale delle olive dal file txt
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as file:
                true_count = int(file.read().strip())
                print(f"Numero di olive REALI per {images}: {count} ")
                true_counts.append(true_count)
        else:
            print(f"Warning: {txt_file} non trovato, salto il calcolo del true count per questa immagine")

    # Calcola MSE e deviazione standard
    mse, std_dev = compute_mse_and_devStd(true_counts, predicted_counts)
    print(f"MSE: {mse}")
    print(f"Deviazione standard: {std_dev}")


if __name__ == '__main__':
    # Launch Application
    main()