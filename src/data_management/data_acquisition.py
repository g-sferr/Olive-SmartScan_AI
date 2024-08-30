from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision
from tqdm import tqdm

class OliveDatasetLoader(Dataset):
    """Loader for the Olive Dataset, handling images and their corresponding bounding boxes."""

    def __init__(self, data_dir):
        """Initializes the OliveDatasetLoader with the given data directory.
        
        Args:
            data_dir (str): The directory where the dataset is stored.
        """
        self.data_dir = data_dir
        self.images = []  # List of images in the dataset
        self.bboxes = []  # List of bounding boxes corresponding to each image
        # self.confs = []
        #self._load_data()

    def _load_data(self, subFolder):
        """Loads image file names from the specified subfolder.
        
        Args:
            subFolder (str): The subfolder within the dataset directory containing the images.
        
        Returns:
            list: A list of image file names in the specified subfolder.
        """
        base_dir = os.path.abspath(self.data_dir)
        images_dir = os.path.join(base_dir, subFolder)
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        return image_files
    
        '''
            total_images = len(image_files)
            
            # Usa tqdm per la progress bar
            for image_file in tqdm(image_files, desc="Loading dataset", unit="image"):
                    image_path = os.path.join(images_dir, image_file)
                    labels_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

                    # Carica l'immagine
                    image = Image.open(image_path).convert('RGB')
                    image = self._transform_image(image)
                    self.images.append(image)

                    # Carica l'annotazione
                    #print(f"NomeFile-> {image_file}")
                    bboxes = self._load_labels(labels_file) # Carico le labels di quella specifica immagine
                    self.bboxes.append(bboxes)
                    # self._class.append(_class)
        '''

    def _load_labels(self, subFolder, fileName):
        """Loads bounding boxes and classes from a label file.
        
        Args:
            subFolder (str): The subfolder containing the label file.
            fileName (str): The name of the label file.
        
        Returns:
            tuple: A tuple containing a tensor of classes and a tensor of bounding boxes.
        """
        bboxes = []
        classes = []

        path = self.data_dir + '/' + subFolder + '/' + fileName
        with open(path, 'r') as file:
            lines = file.readlines()
            noLine = True
            for line in lines:
                try:
                    _class, x1, y1, w, h = map(float, line.strip().split())
                    noLine = False
                except:
                    print(f"File non corretto: {fileName}")
                    break
                bboxes.append([int(_class), x1, y1, w, h])
                classes.append(_class)
            if noLine:
                print(f"noLine --> {fileName}")

        return torch.tensor(classes, dtype=torch.int16), torch.tensor(bboxes, dtype=torch.float16)

    def _transform_image(self, image):
        """Transforms the image for input into a neural network.
        
        The image is resized to 640x640, converted to a tensor, and normalized.
        
        Args:
            image (PIL.Image): The image to be transformed.
        
        Returns:
            torch.Tensor: The transformed image.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((640, 640)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def __len__(self):
        """Returns the number of images in the dataset.
        
        Returns:
            int: The number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Returns an image and its corresponding bounding boxes.
        
        Args:
            idx (int): The index of the image and bounding boxes to retrieve.
        
        Returns:
            tuple: A tuple containing the image and its bounding boxes.
        """
        image = self.images[idx]
        bbox = self.bboxes[idx]
        return image, bbox

    def load_and_resize(self, imageNameFile, sourceFolder, subFolder):
        """Loads and resizes an image if its dimensions are not already 640x640.
        
        Args:
            imageNameFile (str): The name of the image file to resize.
            sourceFolder (str): The folder where the original image is located.
            subFolder (str): The subfolder within the source folder.
        """
        image_path = r'C:/Users/Francesco/Desktop/visualize/' + sourceFolder + '/' + subFolder + '/' + imageNameFile
        image = Image.open(image_path)

        orig_size = image.size
        
        if(orig_size == (640, 640)):
            return
        print(f"Resize eseguito per --> {imageNameFile}")

        resize_transform = torchvision.transforms.Resize((640, 640))
        resized_image = resize_transform(image)

        resized_image.save(r'C:/Users/Francesco/Desktop/visualize/resized/' + sourceFolder + '/' + subFolder + '/' + imageNameFile)

    def getTrueOliveCount(self, pathLabels):
        """Reads the true count of olives from a label file.
        
        Args:
            pathLabels (str): The path to the label file containing the olive count.
        
        Returns:
            int: The actual count of olives as indicated in the label file.
        """
        oliveCount = 0
        with open(pathLabels, 'r') as file:
            oliveCount = int(file.read().strip())
            #for line in file:
                #oliveCount = int(line) #int(file.read().strip())
        return oliveCount


def module_tester():
    """Tests the module's functions, including loading and resizing images."""
    oliveDatasetLoader0 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_0')
    imageList = oliveDatasetLoader0._load_data('train')
    imageList.sort()

    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_0', 'train')


if __name__ == '__main__':
    module_tester()