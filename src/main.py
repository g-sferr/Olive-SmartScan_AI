from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import os
from src.data_management.data_acquisition import OliveDatasetLoader

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
    results = model(img_path) #model.predict(img_path) #
    
    #allBoxes = []
    #allCls = []
    #untouchedBBOX = []
    #untouchedCLS = []
    #for r in results:
    #untouchedBBOX.append(results[0].boxes.xyxy.cpu().numpy())
    #untouchedCLS.append(results[0].boxes.cls.cpu().numpy())
    boxes = results[0].boxes
    boxes = boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()
    #allBoxes.append(boxes)
    #allCls.append(cls)

    # Definire le classi
    TREE_CLASS_ID = 0
    CROWN_CLASS_ID = 1
    OLIVE_CLASS_ID = 2

    # Separare le bounding box in base alle classi
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
    data_dir = r'C:\Users\Francesco\Desktop\tempDatasetOlive'
    oliveDatasetLoader = OliveDatasetLoader(data_dir)
    subFolder = 'FotoDiProvaModelloX'
    dataloader = oliveDatasetLoader._load_data(subFolder)

    # Carica il modello YOLO pre-addestrato
    model = YOLO(r'C:\Users\Francesco\Desktop\ModelloX\best.pt')  # Sostituisci 'best.pt' con il percorso del tuo modello

    # Processa ogni immagine
    for images in dataloader:
        imagePath = os.path.join(data_dir, subFolder)
        imagePath = os.path.join(imagePath, images) 
        count = count_olives(imagePath, model)
        print(f"Numero di olive in {images}: {count} olive in chioma")


if __name__ == '__main__':
    # Launch Application
    main()