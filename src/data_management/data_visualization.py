import os # For os operation
import cv2 # For image processing and OpenCV functionalities
import matplotlib.pyplot as plt


# ******************** BEGIN-Object Detection Seminar Code ********************

"""
The following code, located between the comments "BEGIN-Object Detection Seminar Code" and
"END-Object Detection Seminar Code", serves as supplementary support for various visualization
techniques used during the experimental phase.

Please note that the intellectual property rights for this code belong to Engineer Miglionico,
who provided it during the Object Detection seminar held as part of the Intelligent Systems course
at the University of Pisa.

"""

def conta_istanze_darknet(file_path):
    istanze = {}  # Dictionary to store the counts of instances
    
    # Reading the content of the annotation file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterating through each line in the file
    for line in lines:
        line = line.strip()  # Removing leading and trailing whitespaces
        # Skipping lines starting with '#' or empty lines
        if line.startswith('#') or line == '':
            continue
        
        # Extracting the class label from the line
        classe, *_ = line.split(' ')
        
        try:
            classe = int(classe)  # Converting the class label to an integer
        except ValueError:
            # Handling the case where the class label cannot be converted to an integer
            print(f"Warning: The first value cannot be converted to an integer in the line: {line}")
            continue

        istanza = classe  # Considering each class label as an 'instance'

        # Updating the count of instances for each class label
        if istanza in istanze:
            istanze[istanza] += 1
        else:
            istanze[istanza] = 1

    return istanze  # Returning the dictionary containing the counts of instances for each class label

def analizza_cartella(directory):
    risultati_totali = {}  # Dictionary to store the total counts of instances across all files

    # Iterating through each file in the directory
    for filename in os.listdir(directory):   
        if filename.endswith(".txt"):  # Considering only files with '.txt' extension
            file_path = os.path.join(directory, filename)  # Getting the full path of the file
            istanze_contate = conta_istanze_darknet(file_path)  # Counting instances in the current file

            # Updating the total counts of instances across all files
            for istanza, conteggio in istanze_contate.items():
                if istanza in risultati_totali:
                    risultati_totali[istanza] += conteggio
                else:
                    risultati_totali[istanza] = conteggio

    # Printing the total counts of instances for each class label
    print("\nTotal Results:")
    for istanza, conteggio in risultati_totali.items():
        print(f"{istanza}: {conteggio}")

def draw_bbox(image, class_id, x_center, y_center, width, height, class_map):
    """
    Draws a bounding box around an object in an image and labels it with the class name.

    Parameters:
    - image (numpy.ndarray): The image on which to draw.
    - class_id (int): The ID of the class to which the object belongs.
    - x_center, y_center, width, height (float): The center coordinates, width, and height of the bounding box, relative to the image size.
    - class_map (dict): A mapping from class IDs to class names.

    This function first converts the relative bounding box coordinates to absolute pixel coordinates. 
    It then draws a rectangle (bounding box) on the image and labels it with the corresponding class name.
    """
    # Convert relative coordinates to absolute pixel values based on image dimensions
    x_center, y_center, width, height = (x_center * image.shape[1], y_center * image.shape[0],
                                         width * image.shape[1], height * image.shape[0])

    # Calculate the top-left and bottom-right corners of the bounding box
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Draw the rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add the class name label above the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, class_map.get(class_id, f'classe{class_id}'), (x_min, y_min - 10), font, 0.6, (0, 255, 0), 2)

def process_image(image_path, label_path, class_map):
    """
    Processes an image by displaying bounding boxes and labels as specified in a label file.

    Parameters:
    - image_path (str): The file path of the image to process.
    - label_path (str): The file path of the label file containing bounding box and class ID information.
    - class_map (dict): A mapping from class IDs to class names.

    The function reads an image and its corresponding label file. For each object annotation in the label file,
    it draws a bounding box and labels it. The image is then displayed. The function waits for a key press; pressing
    the space bar continues to the next image, while pressing 'Q' or 'q' exits the processing.
    """
    # Load the image
    image = cv2.imread(image_path)
    # Check if the image was successfully loaded
    if image is None:
        print(f"Errore nel caricamento dell'immagine {image_path}")
        return False

    # Process the label file if it exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.split()
                # Ensure the line contains all necessary parts
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Draw the bounding box for each object in the image
                    draw_bbox(image, int(class_id), x_center, y_center, width, height, class_map)

    # Display the processed image
    cv2.imshow(f'{image_path}', image)
    key = cv2.waitKey(0)  # Wait for a key press

    # Space bar continues processing, 'Q' or 'q' quits
    if key == 32:  # ASCII for space bar
        cv2.destroyAllWindows()
        return True
    elif key in [81, 113]:  # ASCII for 'Q' and 'q'
        cv2.destroyAllWindows()
        return False
    
def process_directory(directory, class_map):
    """
    Processes all images in a directory, displaying bounding boxes and labels for each.

    Parameters:
    - directory (str): The path to the directory containing the images and label files.
    - class_map (dict): A mapping from class IDs to class names.

    This function iterates over all image files in the specified directory, processes each image
    using the process_image function, and displays them with their bounding boxes and labels. 
    The user can navigate through the images using the space bar and exit the loop with 'Q' or 'q'.
    """
    # Iterate over all files in the directory
    for filename in sorted(os.listdir(directory)):
        # Check for image files based on their extensions
        if filename.endswith('.jpg') or filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            label_path = os.path.splitext(image_path)[0] + '.txt'
            # Process each image and decide whether to continue based on user input
            continue_processing = process_image(image_path, label_path, class_map)
            if not continue_processing:
                break

# ******************** END-Object Detection Seminar Code ********************


def plot_errors(true_counts, predicted_counts):
    """
    Plots a histogram of the errors between true counts and predicted counts.

    Parameters:
    - true_counts: List of actual counts.
    - predicted_counts: List of predicted counts by the model.

    The function calculates the error for each pair of true and predicted counts,
    and then plots the distribution of these errors using a histogram.
    """
    errors = [true - pred for true, pred in zip(true_counts, predicted_counts)]
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Error (True Count - Predicted Count)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Counting Errors')
    plt.show()

def draw_bbox_from_model(imageCV2, class_id, box, class_map):
    """
    Draws a bounding box on an image with a label indicating the class.

    Parameters:
    - imageCV2: The image on which to draw the bounding box (in OpenCV format).
    - class_id: The ID of the class for this bounding box.
    - box: A tuple containing normalized coordinates (x_min, y_min, x_max, y_max).
    - class_map: A dictionary mapping class IDs to class names.

    The function converts normalized bounding box coordinates to pixel coordinates
    and then draws the box on the image with the corresponding class label.
    """
    image_height, image_width = imageCV2.shape[:2]
    normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max = box

    # Convert normalized coordinates to absolute pixel values based on image dimensions
    x_min = int(normalized_x_min * image_width)
    y_min = int(normalized_y_min * image_height)
    x_max = int(normalized_x_max * image_width)
    y_max = int(normalized_y_max * image_height)

    # Draw the rectangle on the image
    cv2.rectangle(imageCV2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # Add the class name label above the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imageCV2, class_map.get(class_id, f'class{class_id}'),
                (x_min, y_min - 10), font, 0.6, (0, 255, 0), 2)

def module_tester():
    # Code for test functions of the module
    
    # ***** Drawn BBox on Image *****
    
    # Mapping from class IDs to class names
    class_map = {
        0: "tree",
        1: "crown",
        2: "olive"
    }
    # Path to the directory containing the images and label files
    directory_path = r'C:/path/with/image/and/labels/file'
    # Start processing the directory
    process_directory(directory_path, class_map)
    

    # ***** Counting of Classes *****

    # Path to the directory containing the dataset
    #cartella_dataset = r"C:/path/with/image/and/labels/file"
 
    #if os.path.exists(cartella_dataset):
    #    analizza_cartella(cartella_dataset)  # Analyzing the dataset directory and counting classes
    #else:
    #    print(f"The directory {cartella_dataset} does not exist.")


if __name__ == '__main__':
    module_tester()