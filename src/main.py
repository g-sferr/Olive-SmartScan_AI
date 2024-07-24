
import cv2
import multiprocessing
import os
import torch
import numpy as np
import torch.nn as nn
from math import sqrt
from statistics import stdev
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import mean_squared_error
from src.data_management.data_acquisition import OliveDatasetLoader
from src.data_management.data_visualization import draw_bbox_from_model

# Classi
TREE_CLASS_ID = 0
CROWN_CLASS_ID = 1
OLIVE_CLASS_ID = 2

class_map = {
    TREE_CLASS_ID: "tree",
    CROWN_CLASS_ID: "crown",
    OLIVE_CLASS_ID: "olive"
    }

def getTrueOliveCount(pathLabels):
    oliveCount = 0
    with open(pathLabels, 'r') as file:
        oliveCount = int(file.read().strip())
        #for line in file:
            #oliveCount = int(line) #int(file.read().strip())
    return oliveCount

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

    imageCV2 = cv2.imread(img_path)
    if imageCV2 is None:
        raise ValueError(f"Could not load image {img_path}")
    
    image_height, image_width = imageCV2.shape[:2]
    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        normalized_x_min = x_min / image_width
        normalized_y_min = y_min / image_height
        normalized_x_max = x_max / image_width
        normalized_y_max = y_max / image_height
        normalized_boxes.append([normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max])

    # Separazione bounding box in base alle classi
    tree_boxes = [box for i, box in enumerate(normalized_boxes) if cls[i] == TREE_CLASS_ID]
    crown_boxes = [box for i, box in enumerate(normalized_boxes) if cls[i] == CROWN_CLASS_ID]
    olive_boxes = [box for i, box in enumerate(normalized_boxes) if cls[i] == OLIVE_CLASS_ID]
    
    oliveOnTree = 0
    for olive_box in olive_boxes:
        for crown_box in crown_boxes:
            draw_bbox_from_model(imageCV2, CROWN_CLASS_ID, crown_box, class_map)
            if is_contained(olive_box, crown_box):
                if tree_boxes: # Verifica se la chioma è contenuta nell'albero
                    for tree_box in tree_boxes:
                        draw_bbox_from_model(imageCV2, TREE_CLASS_ID, tree_box, class_map)
                        if is_contained(crown_box, tree_box):
                            oliveOnTree += 1
                            draw_bbox_from_model(imageCV2, OLIVE_CLASS_ID, olive_box, class_map)
                            break  # Esci dal loop dopo aver trovato un albero contenente la chioma
                else:
                    # Se non ci sono alberi, conta comunque l'oliva
                    oliveOnTree += 1
                    draw_bbox_from_model(imageCV2, OLIVE_CLASS_ID, olive_box, class_map)

    oliveOutTree = len(olive_boxes) - oliveOnTree

    assert(len(olive_boxes) == oliveOutTree + oliveOnTree)

    return oliveOnTree, oliveOutTree, imageCV2

def plot_errors(true_countsOnTree, predicted_countsOnTree, true_countsOutTree, predicted_countsOutTree):
    errors = [true - pred for true, pred in zip(true_countsOnTree, predicted_countsOnTree)]
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Error (True Count - Predicted Count)')
    plt.ylabel('Frequency')
    plt.title('Distribution of On Tree Counting Errors')

    plt.show()

    errors = [true - pred for true, pred in zip(true_countsOutTree, predicted_countsOutTree)]
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Error (True Count - Predicted Count)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Out Tree Counting Errors')

    plt.show()

def main():
    #Code for test functions of the module
    data_dir = r'C:\Users\Francesco\Desktop\countingTest\TrueCount'
    oliveDatasetLoader = OliveDatasetLoader(data_dir)
    subFolder1 = 'oliveConCrown'
    oliveConCrownLoader = oliveDatasetLoader._load_data(subFolder1)

    subFolder2 = 'oliveSenzaCrown'
    oliveSenzaCrownLoader = oliveDatasetLoader._load_data(subFolder2)

    # Carica il modello YOLO pre-addestrato
    model = YOLO(r'C:\Users\Francesco\Desktop\Final_Trained_Models\2_YOLOv8 Small\Best_YOLOv8_S.pt')
    # NANO: MSE: 1085.1048387096773
    # NANO: Deviazione standard: 32.9409295362119
    # SMALL: MSE: 853.7258064516129
    # SMALL: Deviazione standard: 29.21858666074752
    # MEDIUM: 880.1774193548387
    # MEDIUM: Deviazione standard: 29.66778420028767
    # LARGE: MSE: 806.5
    # LARGE: Deviazione standard: 28.398943642325854
    # X: MSE: 889.5806451612904
    # X: Deviazione standard: 29.825838549172264

    true_olive_counts = []
    predicted_olive_counts = []
    # Processa ogni immagine
    for image in oliveConCrownLoader:
        imagePath = os.path.join(data_dir, subFolder1)
        imagePath = os.path.join(imagePath, image) 

        # Aggiunge il conteggio stimato
        pred_olive_on_tree, _ , imageCV2 = count_olives(imagePath, model)
        predicted_olive_counts.append(pred_olive_on_tree)

        true_olive_count = getTrueOliveCount(os.path.join(data_dir + '\\' + subFolder1, str(image).replace(".jpg", "Count.txt")))
        true_olive_counts.append(true_olive_count)

        print(f"{image} CROWN -> Olive pred Tree: {pred_olive_on_tree} | Olive REALI: {true_olive_count}")
        
        #cv2.imshow(imagePath, imageCV2)
        #key = cv2.waitKey(0)  # Wait for a key press
        #cv2.destroyAllWindows()

    # Calcola MSE e deviazione standard
    mse, std_dev = compute_mse_and_devStd(true_olive_counts, predicted_olive_counts)
    print(f"MSE ON-TREE: {mse}")
    print(f"Deviazione standard ON-TREE: {std_dev}")

    pred_olives_out_tree = []
    true_olives_out_tree = []
    for image in oliveSenzaCrownLoader:
        imagePath = os.path.join(data_dir, subFolder2)
        imagePath = os.path.join(imagePath, image) 

        # Aggiunge il conteggio stimato
        _ , pred_olive_out_tree, imageCV2 = count_olives(imagePath, model)
        pred_olives_out_tree.append(pred_olive_out_tree)

        true_olive_count = getTrueOliveCount(os.path.join(data_dir + '\\' + subFolder2, str(image).replace(".jpg", "Count.txt")))
        true_olives_out_tree.append(true_olive_count)

        print(f"{image} SENZA -> | Olive pred Out Tree: {pred_olive_out_tree} | Olive REALI: {true_olive_count}")
        
        #cv2.imshow(imagePath, imageCV2)
        #key = cv2.waitKey(0)  # Wait for a key press
        #cv2.destroyAllWindows()

    # Calcola MSE e deviazione standard
    mse, std_dev = compute_mse_and_devStd(true_olives_out_tree, pred_olives_out_tree)
    print(f"MSE OUT-TREE: {mse}")
    print(f"Deviazione standard OUT-TREE: {std_dev}")

    plot_errors(true_olive_counts, predicted_olive_counts, true_olives_out_tree, pred_olives_out_tree)
    

if __name__ == '__main__':
    # Launch Application
    main()