#Codice per il Renaming dei file

import os
import shutil


def renameFile():
    directory = os.path.abspath('datasets/processed/train_set')
    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')

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

        if img_name.__contains__("_olive"):
            continue

        new_img_name = os.path.join(images_dir, f"{numeric_name}_olive{img_extension}")
        new_lbl_name = os.path.join(labels_dir, f"{numeric_name}_olive{lbl_extension}")

        os.rename( os.path.join(images_dir, image_files[i]), new_img_name)
        os.rename( os.path.join(labels_dir, label_files[i]), new_lbl_name)
        numeric_name += 1
        #print(f"new_img_name: {new_img_name} | new_lbl_name: {new_lbl_name}")

def copyOnlyLabelsFromImages():
    destinationDir = os.path.abspath('datasets/processed/train_set')
    images_dest_dir = os.path.join(destinationDir, 'images')
    labels_dest_dir = os.path.join(destinationDir, 'destLbl')

    sourceDir = os.path.abspath('datasets/processed/train_set/')
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



if __name__ == '__main__':
    #copyOnlyLabelsFromImages()
    renameFile()
    #addCrownBBox()
    print("OK")
