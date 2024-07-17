#Codice per il Renaming dei file

import os
import shutil
import random


def renameFile():
    directory = os.path.abspath('datasets/full_dataset')
    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')

    destDirectory = os.path.abspath('datasets/tempDataset')
    destImagesDir = os.path.join(destDirectory, 'images')
    destlblDir = os.path.join(destDirectory, 'labels')

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    image_files.sort()
    label_files.sort()

    print(f"Length of images: {len(image_files)}")
    print(f"Length of labels: {len(label_files)}")
    numeric_name = 0

    for i in range(len(image_files)):
        img_name, img_extension = os.path.splitext(image_files[i])
        lbl_name, lbl_extension = os.path.splitext(label_files[i])
        print(f"img_name: {img_name} | lbl_name: {lbl_name} ")
        assert (img_name == lbl_name)

        splittedName = img_name.split("_") # Esempio di file: 0_olive.jpg oppure 0_tree.jpg
        typeImg = str(splittedName[1]) # qui memorizzo solamente 'olive' oppure 'tree'

        #if img_name.__contains__("_tree"):
        #    continue

        new_img_name = os.path.join(destImagesDir, f"{numeric_name}_" + typeImg + f"{img_extension}")
        new_lbl_name = os.path.join(destlblDir, f"{numeric_name}_" + typeImg + f"{lbl_extension}")

        print(f"new_img_name: {new_img_name} | new_lbl_name: {new_lbl_name}")
        os.rename( os.path.join(images_dir, image_files[i]), new_img_name)
        os.rename( os.path.join(labels_dir, label_files[i]), new_lbl_name)
        numeric_name += 1
        

def copyOnlyLabelsFromImages():
    destinationDir = os.path.abspath('datasets/processed/DatasetSoloAlberi')
    images_dest_dir = os.path.join(destinationDir, 'images')
    labels_dest_dir = os.path.join(destinationDir, 'destLabels')

    sourceDir = os.path.abspath('datasets/processed/DatasetSoloAlberi')
    source_labels_dir = os.path.join(sourceDir, 'labels')

    image_files = [f for f in os.listdir(images_dest_dir) if f.endswith('.jpg') or f.endswith('.png')]

    label_files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]

    for i in range(len(image_files)):
        img_name, img_extension = os.path.splitext(image_files[i])
        for j in range(len(label_files)):
            lbl_name, lbl_extension = os.path.splitext(label_files[j])
            if lbl_name == img_name:
                shutil.copy( os.path.join(source_labels_dir, label_files[j]), os.path.join(labels_dest_dir, label_files[j]))
                print(f"Copying {img_name}")

def addCrownBBox():
    directory = os.path.abspath('datasets/processed/train_set')
    labels_dir = os.path.join(directory, 'destLbl')
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    crownBBox = "1 0.5 0.5 0.1 0.1"
    for i in range(len(label_files)):
        with open(os.path.join(labels_dir, label_files[i]), 'a') as file:
            file.write('\n' + crownBBox)
            print(f"crownBBox added -> {label_files[i]}")

def changeIDLablesOlive():
    baseDir = os.path.abspath('datasets/processed/train_set')
    labels_dir = os.path.join(baseDir, 'labels')

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    for label_file in label_files:
        file_path = os.path.join(labels_dir, label_file)
        
        print("File com'era:")
        with open(file_path, 'r') as file:
            lines = file.readlines()
            print(lines)
        
        # Modifica le righe che iniziano per '0 '
        modified_lines = ['2' + line[1:] if line.startswith('0 ') else line for line in lines]
        
        print("File com'è adesso:")
        with open(file_path, 'w') as file:
            print(modified_lines)
            file.writelines(modified_lines)


def createShuffledKFold():
    directoryFullDataset = os.path.abspath('datasets/full_dataset')
    imagesFullDatasetDir = os.path.join(directoryFullDataset, 'images')
    labelsFullDatasetDir = os.path.join(directoryFullDataset, 'labels')

    partitionFolderDir = os.path.abspath('datasets/partition_folders')

    imagesFullDataset = [f for f in os.listdir(imagesFullDatasetDir) if f.endswith('.jpg') or f.endswith('.png')]
    labelsFullDataset = [f for f in os.listdir(labelsFullDatasetDir) if f.endswith('.txt')]

    imagesFullDataset.sort()

    #imagesTreeFullDataset = [f for f in os.listdir(imagesFullDatasetDir) if f.endswith('_tree.jpg') or f.endswith('_tree.png')]
    #labelsTreeFullDataset = [f for f in os.listdir(labelsFullDatasetDir) if f.endswith('_tree.txt')]

    foldSize = 200 # maxRandomNumber / 5
    uniqueOliveNumber = []
    uniqueTreeNumber = []
    print(f"foldSize: {2 * foldSize}")

    for i in range(0, 5, 1): # ForEach Fold [0...4]
        for j in range(0, foldSize, 1): # Devo generare 400 numeri casuali DISTINTIù
            fold_iDir = os.path.join(partitionFolderDir, ('fold_' + str(i)))

            while True:
                numeroOlivacasuale = random.randint(0, 1998)
                if (numeroOlivacasuale not in uniqueOliveNumber) and (numeroOlivacasuale % 2 == 0):
                    break
            uniqueOliveNumber.append(numeroOlivacasuale)

            while True:
                numeroTreecasuale = random.randint(1, 1999)
                if (numeroTreecasuale not in uniqueTreeNumber) and (numeroTreecasuale % 2 != 0):
                    break
            uniqueTreeNumber.append(numeroTreecasuale)
            
            oliveFileName = str(numeroOlivacasuale) + "_olive"

            shutil.copy( os.path.join(imagesFullDatasetDir, oliveFileName + ".jpg"), os.path.join(fold_iDir, oliveFileName + ".jpg"))
            shutil.copy( os.path.join(labelsFullDatasetDir, oliveFileName + ".txt"), os.path.join(fold_iDir, oliveFileName + ".txt"))

            treeFilename = str(numeroTreecasuale) + "_tree"

            shutil.copy( os.path.join(imagesFullDatasetDir, treeFilename + ".jpg"), os.path.join(fold_iDir, treeFilename + ".jpg"))
            shutil.copy( os.path.join(labelsFullDatasetDir, treeFilename + ".txt"), os.path.join(fold_iDir, treeFilename + ".txt"))
            
            #print(f"\tCopiato il file: {imagesOliveFullDataset[numero_casuale]} + {labelsOliveFullDataset[numero_casuale]}")
        
        print(f"Fold {i}: Filled")



if __name__ == '__main__':
    #copyOnlyLabelsFromImages()
    #renameFile()
    #addCrownBBox()
    #changeIDLablesOlive()

    createShuffledKFold()
    print("OK")
