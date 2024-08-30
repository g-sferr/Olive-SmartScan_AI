import cv2
import torch
from math import sqrt
from statistics import stdev
from sklearn.metrics import mean_squared_error
from src.data_management.data_visualization import draw_bbox_from_model


def compute_mse_and_devStd(true_counts, predicted_counts):
    """Computes the Mean Squared Error (MSE) and Root Mean Square Deviation (RMSD) 
    between the true and predicted counts of olives.

    Args:
        true_counts (list): The actual number of olives.
        predicted_counts (list): The predicted number of olives by the model.

    Returns:
        tuple: A tuple containing the MSE and RMSD values.
    """
    mse = mean_squared_error(true_counts, predicted_counts)
    rmsd = sqrt(mse)

    return mse, rmsd

def is_contained(box1, box2):
    """Checks whether a bounding box (box1) is completely contained 
    within another bounding box (box2).

    Args:
        box1 (list): Coordinates of the first bounding box (x1, y1, x2, y2).
        box2 (list): Coordinates of the second bounding box (x1, y1, x2, y2).

    Returns:
        bool: True if box1 is completely contained within box2, False otherwise.
    """
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and
            box1[2] <= box2[2] and box1[3] <= box2[3])

def count_olives(img_path, model):
    """Counts the number of olives on and off the tree by analyzing the bounding boxes
    predicted by the YOLO model.

    Args:
        img_path (str): Path to the image file.
        model (torch.nn.Module): The trained YOLO model used for detection.

    Returns:
        tuple: A tuple containing the number of olives on the tree, off the tree,
               and the processed image with bounding boxes drawn.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = model(img_path)
    
    # Extract bounding boxes and class labels from the YOLO model results
    boxes = results[0].boxes
    boxes = boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()

    # Load the image using OpenCV
    imageCV2 = cv2.imread(img_path)
    if imageCV2 is None:
        raise ValueError(f"Could not load image {img_path}")
    
    image_height, image_width = imageCV2.shape[:2]

    # Normalize the bounding box coordinates relative to the image dimensions
    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        normalized_x_min = x_min / image_width
        normalized_y_min = y_min / image_height
        normalized_x_max = x_max / image_width
        normalized_y_max = y_max / image_height
        normalized_boxes.append([normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max])

    # Define the class IDs for Tree, Crown, and Olive
    TREE_CLASS_ID = 0
    CROWN_CLASS_ID = 1
    OLIVE_CLASS_ID = 2

    # Map class IDs to their corresponding labels
    class_map = {
        TREE_CLASS_ID: "tree",
        CROWN_CLASS_ID: "crown",
        OLIVE_CLASS_ID: "olive"
    }

    # Separate bounding boxes based on their class labels
    tree_boxes = [box for i, box in enumerate(normalized_boxes) if cls[i] == TREE_CLASS_ID]
    crown_boxes = [box for i, box in enumerate(normalized_boxes) if cls[i] == CROWN_CLASS_ID]
    olive_boxes = [box for i, box in enumerate(normalized_boxes) if cls[i] == OLIVE_CLASS_ID]
    
    oliveOnTree = 0
    for olive_box in olive_boxes:
        for crown_box in crown_boxes:
            draw_bbox_from_model(imageCV2, CROWN_CLASS_ID, crown_box, class_map)
            if is_contained(olive_box, crown_box):
                if tree_boxes:  # Check whether the crown is contained within a tree
                    for tree_box in tree_boxes:
                        draw_bbox_from_model(imageCV2, TREE_CLASS_ID, tree_box, class_map)
                        if is_contained(crown_box, tree_box):
                            oliveOnTree += 1
                            draw_bbox_from_model(imageCV2, OLIVE_CLASS_ID, olive_box, class_map)
                            break  # Exit the loop after finding a tree containing the crown
                else:
                    # If there are no trees, the olive still counts as "on tree"
                    oliveOnTree += 1
                    draw_bbox_from_model(imageCV2, OLIVE_CLASS_ID, olive_box, class_map)

    oliveOutTree = len(olive_boxes) - oliveOnTree

    # Ensure the total number of olives equals the sum of "on tree" and "off tree"
    assert(len(olive_boxes) == oliveOutTree + oliveOnTree)

    return oliveOnTree, oliveOutTree, imageCV2


def module_tester():
    # Code for test functions of the module
    return

if __name__ == '__main__':
    module_tester()