from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

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
    results = model(img_path)
    boxes = results.xyxy[0].numpy()
    cls = results.cls[0].numpy()

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
    data_dir = 'path/to/data'
    dataloader = load_data(data_dir)

    # Carica il modello YOLO pre-addestrato
    model = YOLO('best.pt')  # Sostituisci 'best.pt' con il percorso del tuo modello


    # Processa ogni immagine
    for input in dataloader:
        count = count_olives(input, model)
        print(f"Numero di olive in {input}: {count}")


if __name__ == '__main__':
    # Launch Application
    main()