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

    def _load_labels(self, subFolder):
        base_dir = os.path.abspath(self.data_dir)
        labels_dir = os.path.join(base_dir, subFolder)

        labels_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        bboxes = []
        classes = []

        for labels_file in labels_files:
            path = self.data_dir + '/' + subFolder + '/' + labels_file
            with open(path, 'r') as file:
                lines = file.readlines()
                NoLine = True
                for line in lines:
                    try:
                        _class, x1, y1, w, h = map(float, line.strip().split())
                        print(type(x1))
                        NoLine = False
                    except:
                        print(f"File non corretto: {labels_file}")
                        break
                    bboxes.append([int(_class), x1, y1, w, h])
                    classes.append(_class)
                    # _class.append(conf)
                if NoLine:
                    print(f"NoLine --> {labels_file}")

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


def module_tester():
    #Code for test functions of the module

    oliveDatasetLoader0 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_0')
    oliveDatasetLoader1 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_1')
    oliveDatasetLoader2 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_2')
    oliveDatasetLoader3 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_3')
    oliveDatasetLoader4 = OliveDatasetLoader(r'C:\Users\Francesco\Desktop\visualize\ROUND_4')

    imageList = oliveDatasetLoader0._load_data('train')
    imageList.sort()
    incr = imageList[1247]
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_0', 'train')
        input()

    imageList = oliveDatasetLoader0._load_data('test')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_0', 'test')

    imageList = oliveDatasetLoader0._load_data('val')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_0', 'val')
    print("ROUND_0--> OK")

    imageList = oliveDatasetLoader1._load_data('train')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_1', 'train')

    imageList = oliveDatasetLoader1._load_data('test')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_1', 'test')

    imageList = oliveDatasetLoader1._load_data('val')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_1', 'val')
    print("ROUND_1--> OK")

    imageList = oliveDatasetLoader2._load_data('train')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_2', 'train')

    imageList = oliveDatasetLoader2._load_data('test')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_2', 'test')

    imageList = oliveDatasetLoader2._load_data('val')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_2', 'val')
    print("ROUND_2--> OK")

    imageList = oliveDatasetLoader3._load_data('train')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_3', 'train')

    imageList = oliveDatasetLoader3._load_data('test')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_3', 'test')

    imageList = oliveDatasetLoader3._load_data('val')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_3', 'val')
    print("ROUND_3--> OK")

    imageList = oliveDatasetLoader4._load_data('train')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_4', 'train')

    imageList = oliveDatasetLoader4._load_data('test')
    for image in imageList:
        oliveDatasetLoader0.load_and_resize(image, 'ROUND_4', 'test')

    imageList = oliveDatasetLoader4._load_data('val')
    for image in imageList:
            oliveDatasetLoader0.load_and_resize(image, 'ROUND_4', 'val')
    print("ROUND_4--> OK")





if __name__ == '__main__':
    module_tester()