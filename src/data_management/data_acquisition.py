from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision
from tqdm import tqdm

# Esempio di dataset personalizzato
class OliveDatasetLoader(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = [] # images e bboxes sono due vettori Paralleli
        self.bboxes = [] # Quindi bboxes è NECESSARIAMENTE un vettore di vettori (ogni immagine può avere più boundingBoxes)
        # self.confs = []
        #self._load_data()

    def _load_data(self, subFolder):
        base_dir = os.path.abspath(self.data_dir)
        images_dir = os.path.join(base_dir, subFolder)
        #labels_dir = os.path.join(base_dir, 'labels')

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

    def _load_labels(self, subFolder, fileName): # Dato un file, ritorna la lista delle classi e la lista delle bboxes in quel file
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
                # _class.append(conf)
            if noLine:
                print(f"noLine --> {fileName}")

        return torch.tensor(classes, dtype=torch.int16), torch.tensor(bboxes, dtype=torch.float16) # torch.tensor(_class, dtype=torch.float32)

    def _transform_image(self, image):
        # Trasforma l'immagine (ridimensionamento, normalizzazione, ecc.)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((640, 640)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        bbox = self.bboxes[idx]
        # conf = self.confs[idx]
        return image, bbox # conf

    def load_and_resize(self, imageNameFile, sourceFolder, subFolder):

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
        oliveCount = 0
        with open(pathLabels, 'r') as file:
            oliveCount = int(file.read().strip())
            #for line in file:
                #oliveCount = int(line) #int(file.read().strip())
        return oliveCount


def module_tester():

    # Code for test functions of the module, an example below for load_and_resize
    oliveDatasetLoader0 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_0')
    imageList = oliveDatasetLoader0._load_data('train')
    imageList.sort()

    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_0', 'train')


if __name__ == '__main__':
    module_tester()