import os
import numpy as nn
from ultralytics import YOLO
from src.data_management.data_visualization import plot_errors
from src.data_management.data_acquisition import OliveDatasetLoader
from src.inference.utils import compute_mse_and_devStd, count_olives


def main():
    #Code for test functions of the module
    data_dir = r'C:\Users\Francesco\Desktop\countingTest\TrueCount'
    oliveDatasetLoader = OliveDatasetLoader(data_dir)
    subFolder1 = 'oliveConCrown'
    oliveConCrownLoader = oliveDatasetLoader._load_data(subFolder1)

    # Carica il modello YOLO pre-addestrato
    model = YOLO(r'C:\Users\Francesco\Desktop\Final_Trained_Models\5_YOLOv8 XLarge\Best_YOLOv8_X.pt')

    true_olive_counts = []
    predicted_olive_counts = []
    # Processa ogni immagine
    for image in oliveConCrownLoader:
        imagePath = os.path.join(data_dir, subFolder1)
        imagePath = os.path.join(imagePath, image) 

        # Aggiunge il conteggio stimato
        pred_olive_on_tree, _ , imageCV2 = count_olives(imagePath, model)
        predicted_olive_counts.append(pred_olive_on_tree)

        true_olive_count = oliveDatasetLoader.getTrueOliveCount(os.path.join(data_dir + '\\' + subFolder1, str(image).replace(".jpg", "Count.txt")))
        true_olive_counts.append(true_olive_count)

        print(f"{image} CROWN -> Olive pred Tree: {pred_olive_on_tree} | Olive REALI: {true_olive_count}")
        
        # Se si vuole visualizzare il plot, sfruttando "draw_bbox_from_model" delle bboxes togliere commenti qui sotto
        # cv2.imshow(imagePath, imageCV2)
        # key = cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()

    # Calcola MSE e deviazione standard
    mse, std_dev = compute_mse_and_devStd(true_olive_counts, predicted_olive_counts)
    print(f"MSE ON-TREE: {mse}")
    print(f"Deviazione standard ON-TREE: {std_dev}")

    plot_errors(true_olive_counts, predicted_olive_counts)
    

if __name__ == '__main__':
    main()