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
        self._load_data()

    def _load_data(self):
        base_dir = os.path.abspath(self.data_dir)
        images_dir = os.path.join(base_dir, 'images')
        labels_dir = os.path.join(base_dir, 'labels')

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
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

    def _load_labels(self, labels_file):
        bboxes = []
        # _class = []
        with open(labels_file, 'r') as file:
            lines = file.readlines()
            NoLine = True
            for line in lines:
                try:
                    _class, x1, y1, x2, y2 = map(float, line.strip().split())
                    NoLine = False
                except:
                    print(f"File non corretto: {labels_file}")
                    break
                bboxes.append([x1, y1, x2, y2])
                # _class.append(conf)
            if NoLine:
                print(f"NoLine --> {labels_file}")
        return torch.tensor(bboxes, dtype=torch.float32) # torch.tensor(_class, dtype=torch.float32)

    def _transform_image(self, image):
        # Trasforma l'immagine (ridimensionamento, normalizzazione, ecc.)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((648, 648)),
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

def module_tester():
    #Code for test functions of the module
    return

if __name__ == '__main__':
    module_tester()