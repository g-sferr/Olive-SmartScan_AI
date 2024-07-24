import cv2
import torch
from math import sqrt
from statistics import stdev
from sklearn.metrics import mean_squared_error
from src.data_management.data_visualization import draw_bbox_from_model


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

    # Normalizzazzione delle coordinate delle bboxes prima di usarli
    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        normalized_x_min = x_min / image_width
        normalized_y_min = y_min / image_height
        normalized_x_max = x_max / image_width
        normalized_y_max = y_max / image_height
        normalized_boxes.append([normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max])

    # Classes
    TREE_CLASS_ID = 0
    CROWN_CLASS_ID = 1
    OLIVE_CLASS_ID = 2

    class_map = {
    TREE_CLASS_ID: "tree",
    CROWN_CLASS_ID: "crown",
    OLIVE_CLASS_ID: "olive"
    }

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