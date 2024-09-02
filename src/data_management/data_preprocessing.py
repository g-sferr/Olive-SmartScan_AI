import os
import shutil
import random
from src.data_management.data_acquisition import OliveDatasetLoader

def renameFile(workingPath):
    """Renames image and label files in the specified directory with a numeric prefix.
    
    Args:
        workingPath (str): The path to the working directory containing 'images' and 'labels' folders.
    """
    directory = os.path.abspath(workingPath)
    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')

    destDirectory = os.path.abspath(workingPath)
    destImagesDir = os.path.join(destDirectory, 'Rimages')
    destlblDir = os.path.join(destDirectory, 'Rlabels')

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    image_files.sort()
    label_files.sort()

    print(f"Length of images: {len(image_files)}")
    print(f"Length of labels: {len(label_files)}")
    numeric_name = 2000

    for i in range(len(image_files)):
        img_name, img_extension = os.path.splitext(image_files[i])
        lbl_name, lbl_extension = os.path.splitext(label_files[i])
        print(f"img_name: {img_name} | lbl_name: {lbl_name} ")
        assert (img_name == lbl_name)

        #splittedName = img_name.split("_") # Esempio di file: 0_olive.jpg oppure 0_tree.jpg
        #typeImg = str(splittedName[1]) # qui memorizzo solamente 'olive' oppure 'tree'
        typeImg = 'olive'
        #if img_name.__contains__("_tree"):
        #    continue

        new_img_name = os.path.join(destImagesDir, f"{numeric_name}_" + typeImg + f"{img_extension}")
        new_lbl_name = os.path.join(destlblDir, f"{numeric_name}_" + typeImg + f"{lbl_extension}")

        print(f"new_img_name: {new_img_name} | new_lbl_name: {new_lbl_name}")
        os.rename(os.path.join(images_dir, image_files[i]), new_img_name)
        os.rename(os.path.join(labels_dir, label_files[i]), new_lbl_name)
        numeric_name += 2


def copyOnlyLabelsFromImages(path):
    """Copies only label files that correspond to existing image files in the 'images' directory.
    
    Args:
        path (str): The path to the directory containing 'images' and 'labels' folders.
    """
    destinationDir = os.path.abspath(path)
    images_dest_dir = os.path.join(destinationDir, 'images')
    labels_dest_dir = os.path.join(destinationDir, 'labels')

    sourceDir = os.path.abspath(path)
    source_labels_dir = os.path.join(sourceDir, 'labels')

    image_files = [f for f in os.listdir(images_dest_dir) if f.endswith('.jpg')]

    label_files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]

    for i in range(len(image_files)):
        img_name, img_extension = os.path.splitext(image_files[i])
        for j in range(len(label_files)):
            lbl_name, lbl_extension = os.path.splitext(label_files[j])
            if lbl_name == img_name:
                shutil.copy(os.path.join(source_labels_dir, label_files[j]), os.path.join(labels_dest_dir, label_files[j]))
                print(f"Copying {img_name}")


def addCrownBBox(path):
    """Appends a predefined bounding box (crown) to all label files in the directory.
    
    Args:
        path (str): The path to the directory containing the 'labels' folder.
    """
    directory = os.path.abspath(path)
    labels_dir = os.path.join(directory, 'labels')
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    crownBBox = "1 0.5 0.5 1 1" # This represents the CrownBBox, that is a bbox as large as the entire image
    for i in range(len(label_files)):
        with open(os.path.join(labels_dir, label_files[i]), 'a') as file:
            file.write('\n' + crownBBox)
            print(f"CrownBBox added for -> {label_files[i]}")


def changeClassInsideLabelsFile(correctClass, folderBase, subFolder):
    """Changes the class label in each label file to the specified correct class.
    
    Args:
        correctClass (int): The correct class label to set.
        folderBase (str): The base directory containing the subfolder with label files.
        subFolder (str): The subfolder within the base directory containing the label files.
    """
    baseDir = os.path.abspath(folderBase)
    labels_dir = os.path.join(baseDir, subFolder)

    endType = '_tree' if(correctClass == 0) else '_crown' if(correctClass == 1) else '_olive'

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(endType + '.txt')]

    for label_file in label_files:
        file_path = os.path.join(labels_dir, label_file)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        #print(f"File {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        modified_lines = []

        # Modifica le righe che iniziano per '0 '
        for line in lines:
            if not line.startswith(str(correctClass)):
                X = line

        # Modify lines that do not start with the correct class
        modified_lines = [str(correctClass) + line[1:] if (not line.startswith(str(correctClass))) else line for line in lines]
        baseDestDir = os.path.abspath(folderBase)
        labelsDestDir = os.path.join(baseDestDir, 'C' + subFolder)

        file_path = os.path.join(labelsDestDir, label_file)
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)
    print(f"CorrectClass_OK --> {folderBase + '/' + subFolder}")


def createShuffledKFold(sourcePath, destPath):
    """Creates shuffled k-fold partitions of the dataset with a specific distribution of tree and olive images.
    
    Args:
        sourcePath (str): The path to the full dataset directory.
        destPath (str): The path to the destination directory for the k-fold partitions.
    """
    # The FULL_DATASET has 1635 photos of which 1000Trees, 635 Olives (Technically 472lives and 163Olives+Chioma)
    # The distribution, since the model can immediately recognize trees, must predict:
    # -] 200 photos of trees for Fold
    # -] 127 photos of olives for Fold

    directoryFullDataset = os.path.abspath(sourcePath)
    imagesFullDatasetDir = os.path.join(directoryFullDataset, 'images')
    labelsFullDatasetDir = os.path.join(directoryFullDataset, 'labels')

    partitionFolderDir = os.path.abspath(destPath)

    imagesFullDataset = [f for f in os.listdir(imagesFullDatasetDir) if f.endswith('.jpg') or f.endswith('.png')]
    labelsFullDataset = [f for f in os.listdir(labelsFullDatasetDir) if f.endswith('.txt')]

    imagesFullDataset.sort()

    #imagesTreeFullDataset = [f for f in os.listdir(imagesFullDatasetDir) if f.endswith('_tree.jpg') or f.endswith('_tree.png')]
    #labelsTreeFullDataset = [f for f in os.listdir(labelsFullDatasetDir) if f.endswith('_tree.txt')]
    datasetSize = 1635
    treeImagesNumber = 1000
    oliveImageNumber = 635
    assert(datasetSize == (treeImagesNumber + oliveImageNumber))

    uniqueOliveNumber = []
    uniqueTreeNumber = []

    for i in range(0, 5, 1):  # ForEach Fold [0...4]
        fold_iDir = os.path.join(partitionFolderDir, ('fold_' + str(i)))

        for j in range(0, int(oliveImageNumber / 5), 1):
            while True:
                numeroOlivacasuale = random.randint(0, 1268)
                if (numeroOlivacasuale not in uniqueOliveNumber) and (numeroOlivacasuale % 2 == 0):
                    break
            uniqueOliveNumber.append(numeroOlivacasuale)
            oliveFileName = str(numeroOlivacasuale) + "_olive"
            shutil.copy(os.path.join(imagesFullDatasetDir, oliveFileName + ".jpg"), os.path.join(fold_iDir, oliveFileName + ".jpg"))
            shutil.copy(os.path.join(labelsFullDatasetDir, oliveFileName + ".txt"), os.path.join(fold_iDir, oliveFileName + ".txt"))

        print(f"Fold {i}: {int(oliveImageNumber / 5)} Olive selected")

        for j in range(0, int(treeImagesNumber / 5), 1):
            while True:
                numeroTreecasuale = random.randint(1, 1999)
                if (numeroTreecasuale not in uniqueTreeNumber) and (numeroTreecasuale % 2 != 0):
                    break
            uniqueTreeNumber.append(numeroTreecasuale)
            treeFilename = str(numeroTreecasuale) + "_tree"
            shutil.copy(os.path.join(imagesFullDatasetDir, treeFilename + ".jpg"), os.path.join(fold_iDir, treeFilename + ".jpg"))
            shutil.copy(os.path.join(labelsFullDatasetDir, treeFilename + ".txt"), os.path.join(fold_iDir, treeFilename + ".txt"))

        print(f"Fold {i}: {int(treeImagesNumber / 5)} Tree selected")
            #print(f"\tCopiato il file: {imagesOliveFullDataset[numero_casuale]} + {labelsOliveFullDataset[numero_casuale]}")    


def truncate(value, decimal_places=6):
    """Truncates a floating point number to a specified number of decimal places.
    
    Args:
        value (float): The value to be truncated.
        decimal_places (int, optional): The number of decimal places to truncate to. Default is 6.
    
    Returns:
        float: The truncated value.
    """
    factor = 10.0 ** decimal_places
    return int(value * factor) / factor


def truncateBBoxesValues(baseDir, subFolder):
    """Truncates the values of bounding boxes in label files to a specified precision.
    
    Args:
        baseDir (str): The base directory containing the dataset.
        subFolder (str): The subfolder within the base directory containing the label files.
    """
    oliveDatasetLoader = OliveDatasetLoader(baseDir)

    labels_files = [f for f in os.listdir(oliveDatasetLoader.data_dir + '/' + subFolder) if f.endswith('.txt')]
    for label_file in labels_files:
        classesTens, bboxesTens = oliveDatasetLoader._load_labels(subFolder, label_file)
        bboxesList = bboxesTens.tolist()
        classes = classesTens.tolist()
        bboxesAppr = []
        indice = 0
        for bboxes in bboxesList:
            bboxAppr = [truncate(coordinate) for coordinate in bboxes[1:]]
            bboxAppr.insert(0, classes[indice])
            indice += 1
            bboxesAppr.append(bboxAppr)

        with open(oliveDatasetLoader.data_dir + '/' + subFolder + '/truncated/' + label_file, 'w') as file:
            for bboxAppr in bboxesAppr:
                line = " ".join(map(str, bboxAppr))
                line += '\n'
                file.writelines(line)

        print(f"Truncated correctly -> {label_file}")


def getOliveCountFromLabelsFile(pathLabels):
    """Counts the number of olives in a label file based on a specific class label.
    
    Args:
        pathLabels (str): The path to the label file.
    
    Returns:
        int: The count of olives in the label file.
    """
    oliveCount = 0
    with open(pathLabels, 'r') as file:
        for line in file:
            oliveCount = oliveCount + (1 if str(line).startswith("2") else 0)
    return oliveCount


def setOliveNumber(sourcePath):
    """Prompts the user to set the correct number of olives for each image file based on its corresponding label file.
    
    Args:
        sourcePath (str): The path to the directory containing the image and label files.
    """
    file_dir = os.path.abspath(sourcePath)
    images_file = [f for f in os.listdir(file_dir) if f.endswith('.jpg')]
    #labels_file = [f for f in os.listdir(file_dir) if f.endswith('.txt')]

    for image in images_file:
        oliveCount = getOliveCountFromLabelsFile(os.path.join(file_dir, str(image).replace(".jpg", ".txt")))
        correctOliveNumber = input(f"Img: {image} -> | OlivesFromLabels: {oliveCount} | OlivesFromReal: ")
        if correctOliveNumber == '':
            correctOliveNumber = int(oliveCount)

        fileLine = str(correctOliveNumber)
        with open(os.path.join(file_dir, str(image).replace(".jpg", "Count.txt")), 'w') as file:
            file.writelines(fileLine)


def module_tester():
    # Code for test functions of the module, an example below for load_and_resize
    
    setOliveNumber(r'C:/path/with/image/and/labels/file')

    print("OK")


if __name__ == '__main__':
    module_tester()