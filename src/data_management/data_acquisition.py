from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision

# Esempio di dataset personalizzato
class OliveDatasetLoader(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.bboxes = []
        # self.confs = []
        self._load_data()

    def _load_data(self):
        base_dir = os.path.abspath(self.data_dir)
        images_dir = os.path.join(base_dir, 'images')
        annotations_dir = os.path.join(base_dir, 'annotations')

        for image_file in os.listdir(images_dir):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(images_dir, image_file)
                annotation_path = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.txt')

                # Carica l'immagine
                image = Image.open(image_path).convert('RGB')
                image = self._transform_image(image)
                self.images.append(image)

                # Carica l'annotazione
                bboxes = self._load_annotation(annotation_path)
                self.bboxes.append(bboxes)
                # self.confs.append(confs)

    def _transform_image(self, image):
        # Trasforma l'immagine (ridimensionamento, normalizzazione, ecc.)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((648, 648)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def _load_annotation(self, annotation_path):
        bboxes = []
        # confs = []
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                conf, x1, y1, x2, y2 = map(float, line.strip().split())
                bboxes.append([x1, y1, x2, y2])
                # confs.append(conf)
        return torch.tensor(bboxes, dtype=torch.float32) # torch.tensor(confs, dtype=torch.float32)

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