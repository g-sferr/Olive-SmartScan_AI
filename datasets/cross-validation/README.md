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

## Criterion for Cross-Validation Rounds #

- Criteria for inserting the images present in the various *fold_i* into the **test**, **train**, and **val** folders present in the various **ROUND_i** folders within the *"rounds"* folder, representing the various steps of the **cross-validation** operation for **training** and final **evaluation** of the model:

*Notes*: For each round, the **test** folder will contain the complete dataset of one *fold_i* (800) that changes round by round, while **train** and **val** will respectively distribute *80%* (2.560) and *20%* (640) of the images present in all *fold_i - fold_i_in_Test* (round by round as well) excluding the current round's *fold_i*, which is in the **test** folder.

## Cross-Validation Steps in detail:

### ROUND 0 --> Associated YAML

- TEST --> 100% of fold_0
- Train --> 80% of (fold_1, fold_2, fold_3, fold_4)
- VAL --> 20% of (fold_1, fold_2, fold_3, fold_4)

### ROUND 1 --> Associated YAML

- TEST --> 100% of fold_1
- Train --> 80% of (fold_0, fold_2, fold_3, fold_4)
- VAL --> 20% of (fold_0, fold_2, fold_3, fold_4)

### ROUND 2 --> Associated YAML

- TEST --> 100% of fold_2
- Train --> 80% of (fold_1, fold_0, fold_3, fold_4)
- VAL --> 20% of (fold_1, fold_0, fold_3, fold_4)

### ROUND 3 --> Associated YAML

- TEST --> 100% of fold_3
- Train --> 80% of (fold_1, fold_2, fold_0, fold_4)
- VAL --> 20% of (fold_1, fold_2, fold_0, fold_4)

### ROUND 4 --> Associated YAML

- TEST --> 100% of fold_4
- Train --> 80% of (fold_1, fold_2, fold_3, fold_0)
- VAL --> 20% of (fold_1, fold_2, fold_3, fold_0)
