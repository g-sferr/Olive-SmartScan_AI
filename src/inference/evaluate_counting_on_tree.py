import os
import numpy as nn
from ultralytics import YOLO
from src.data_management.data_visualization import plot_errors
from src.data_management.data_acquisition import OliveDatasetLoader
from src.inference.utils import compute_mse_and_devStd, count_olives


def main():
    """Main function to test and evaluate the counting  On-Tree performance of the System.
    
    Loads a dataset of olive images, applies a pre-trained YOLO model to count 
    olives on trees, compares predicted counts with actual counts, and calculates 
    the Mean Squared Error (MSE) and standard deviation.
    """
    data_dir = r'C:/path/with/image/and/labels/for/GroundTruth/TrueCount'
    oliveDatasetLoader = OliveDatasetLoader(data_dir)
    subFolder1 = 'oliveConCrown'
    oliveConCrownLoader = oliveDatasetLoader._load_data(subFolder1)

    # Load the pre-trained YOLO model
    model = YOLO(r'C:/path/containing/pre-trained/detection-model/model_name.pt')

    true_olive_counts = []
    predicted_olive_counts = []

    # Process each image in the dataset
    for image in oliveConCrownLoader:
        imagePath = os.path.join(data_dir, subFolder1)
        imagePath = os.path.join(imagePath, image) 

        # Add the estimated (predicted) count of olives on the tree
        pred_olive_on_tree, _, imageCV2 = count_olives(imagePath, model)
        predicted_olive_counts.append(pred_olive_on_tree)

        # Load the actual count of olives from the corresponding file
        true_olive_count = oliveDatasetLoader.getTrueOliveCount(os.path.join(data_dir + '\\' + subFolder1, str(image).replace(".jpg", "Count.txt")))
        true_olive_counts.append(true_olive_count)

        print(f"{image} Olives predicted On-Tree: {pred_olive_on_tree} | Olive's Actual Number: {true_olive_count}")
        
        # Uncomment below lines to visualize the bounding boxes using cv2
        # cv2.imshow(imagePath, imageCV2)
        # key = cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()

    # Calculate MSE and standard deviation for the predicted versus actual counts
    mse, std_dev = compute_mse_and_devStd(true_olive_counts, predicted_olive_counts)
    print(f"MSE ON-TREE: {mse}")
    print(f"Deviazione standard ON-TREE: {std_dev}")

    plot_errors(true_olive_counts, predicted_olive_counts)
    

if __name__ == '__main__':
    main()