# K-Fold Cross-Validation - K = 5

## DATASET: (1000 Olives + 1000 Trees) 2000 images --> divided into K = 5 folders (0,1,2,3,4) --> 400 images per folder #

- Construction of fold_i in "folders" (i=K):

    **fold_i** = len(dataset_1)/i + len(dataset_2)/i ---> 400 images

    Steps (for each single dataset):
    1. Shuffle the dataset from which you want to extract the *number of images* to be inserted into each *fold_i.*
    2. *Select* the number of images of interest (in our case 200).
    3. Insert the images and their corresponding *labels.txt* files into the *fold_i* folder.

    _Repeat for each dataset_i and fold_i if you have separate datasets of different classes._

At the end, in our case, you will have 5 *fold_i* folders (i=0,...,4) containing 200 olive images and 200 tree images with their respective labels, so the total for each fold is 800 elements.

