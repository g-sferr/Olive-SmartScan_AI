import os
import numpy as nn
from ultralytics import YOLO
from src.data_management.data_visualization import plot_errors
from src.data_management.data_acquisition import OliveDatasetLoader
from src.inference.utils import compute_mse_and_devStd, count_olives


def main():
    """Main function to test and evaluate the counting  Off-Tree performance of the System.
    
    Loads a dataset of olive images, applies a pre-trained YOLO model to 
    count olives outside of trees, compares predicted counts with actual counts,
    and calculates the Mean Squared Error (MSE) and standard deviation.
    """
    data_dir = r'C:\Users\Francesco\Desktop\countingTest\TrueCount'
    oliveDatasetLoader = OliveDatasetLoader(data_dir)
    subFolder2 = 'oliveSenzaCrown'
    oliveSenzaCrownLoader = oliveDatasetLoader._load_data(subFolder2)

    # Load the pre-trained YOLO model
    model = YOLO(r'C:\Users\Francesco\Desktop\Final_Trained_Models\2_YOLOv8 Small\Best_YOLOv8_S.pt')

    pred_olives_out_tree = []
    true_olives_out_tree = []
    
    for image in oliveSenzaCrownLoader:
        imagePath = os.path.join(data_dir, subFolder2)
        imagePath = os.path.join(imagePath, image) 

        # Estimate the count of olives outside the tree
        _ , pred_olive_out_tree, imageCV2 = count_olives(imagePath, model)
        pred_olives_out_tree.append(pred_olive_out_tree)

        # Load the actual count of olives
        true_olive_count = oliveDatasetLoader.getTrueOliveCount(os.path.join(data_dir + '\\' + subFolder2, str(image).replace(".jpg", "Count.txt")))
        true_olives_out_tree.append(true_olive_count)

        print(f"{image} SENZA -> | Olive pred Out Tree: {pred_olive_out_tree} | Olive REALI: {true_olive_count}")
        
        # Uncomment below lines to visualize the bounding boxes using cv2
        # cv2.imshow(imagePath, imageCV2)
        # key = cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()

    # Calculate MSE and standard deviation
    mse, std_dev = compute_mse_and_devStd(true_olives_out_tree, pred_olives_out_tree)
    print(f"MSE OUT-TREE: {mse}")
    print(f"Deviazione standard OUT-TREE: {std_dev}")

    plot_errors(true_olives_out_tree, pred_olives_out_tree)
    

if __name__ == '__main__':
    main()