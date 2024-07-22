
# vOPA_Shigella

This repository contains the information relative to the paper [**Deep learning for classifying anti-*Shigella* opsono-phagocytosis-promoting monoclonal antibodies**]() published in the MICCAI workshop [MOVI](https://sites.google.com/view/movi2024).
The code in this repository can be used to train the model, use it on inference on new data, and obtain the figures in the paper.

## Installation

The libraries needed are in the env.yml file. You can create a conda environment running the command:
```
conda env create -f env.yml
```
The environment can then be activated running the command:
```
conda activate vOPA_Shigella
```

To clone the repository you can run the command:
```
git clone https://github.com/dasch-lab/vOPA_Shigella.git
```

## Dataset
The dataset used in this paper is prioprietary, we will add a link for the download once it is published.

## src/ folder

The src/ contains a script for the training of the model and a script to use the model in inference on new data.
It also contains two folders models/ and utils/, that contain the architectures of the models used and utility functions.

### Training
The training.py file can be used to train the model. There are 3 required arguments to make the script work: the path to the csv with the metadata, the path where the images are stored, and finally the path where to save the results.

To train the model you can run:
```
python src/training.py --dataset_path path/to/dataset --training_dataset path/to/training/dataset --results path/to/results
```

The first experiment shown in the paper can be reproduced running the training with all the possible combinations of the ```--split```, ```--split_type```, and ```--model``` arguments.

The second experiment can be reproduced running the training with all the possible values of the argument ```--concentration_th``` (0.019, 0.039, 0.078, 0.156, 0.312, 0.625, 1.25, 2.5, 5, and 10).

Finally the third experiment can be run changing the value of the ```--channels``` argument, using all possible subsets of the four channels.

### Inference

The inference.py file can be used to classify new images. It requires four arguments: the test set to use, the checkpoint, the path to the images, and where to save the results.

It can be run with:
```
python inference.py --test_set path/to/test --checkpoint path/to/checkpoint --dataset_path path/to/images --output path/to/results
```

## data/ folder
The data folder contains the complete dataset (complete.csv), also divided in training and inference files (train.csv and inference.csv).
It also contains files with the results of the 3 experiments shown in the paper, the results obtained using the model in inference on new data, and finally the data used to create supplementary figure S1.

## Notebook
The ```figures.ipynb``` notebook contains the cells that are needed to create the figures in the main paper (Fig. 1, 2, and 3), and the figure in the supplementary material (Fig. S1).

All the figures can be recreated starting from the data in the data/ folder.

## plots/ folder
The plots folder contains the figures shown in the main paper, that can be recreated with the ```figures.ipynb``` notebook.


## best_efficientnet_s.pt
This is the checkpoint that was used to prove that the model could be used to screen new mAbs and new strains of *Shigella*.
