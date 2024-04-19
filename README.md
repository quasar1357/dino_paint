# dino_paint

dino_paint is an implementation of semantic segmentation using Meta AI's [DINOv2](https://github.com/facebookresearch/dinov2) for feature extraction and a random forest algorithm for classification. It is based on [conv_paint](https://github.com/guiwitz/napari-convpaint) by Guillaume Witz (original idea by Lucien Hinderling), which uses VGG16 for feature extraction instead. The latest version of dino_paint allows combination of any DINOv2 model with any selection of layers of the VGG16 network.

If you decide to use some of my code in any sort of public work, please do contact and cite me.

Once you're here, also have a look at my project [scribbles_creator](https://github.com/quasar1357/scribbles_creator), which can be a great help in testing tools for semantic segmentation such as dino_paint.

## Installation
You can install dino_paint via pip using

    pip install git+https://github.com/quasar1357/dino_paint.git

After this, you can simply import the functions needed in Python (e.g. from dino_paint_utils import selfpredict_dino_forest).


## Main notebook

The notebook [dino_paint.ipynb](dino_paint.ipynb) guides through standard applications such as feature extraction, training and prediciton. It also includes self-prediction, where features of an image are only extracted once, and used for training and prediction on itself. Finally, it enables running automated tests of accuracy and execution times using different model selections.

Use this notebook to learn about how to use the core functions of [dino_paint_utils.py](dino_paint_utils.py) (see below) such as train_dino_forest(), predict_dino_forest() and self_pred_dino_forest().

The code uses some utility and wrapper functions from conv_paint, which is, therefore, a requirement.

Note that the Notebook can also be loaded as a Google Colab notebook by clicking on the badge in its head.

## Utils script

The script [dino_paint_utils.py](dino_paint_utils.py) contains all the functions used for feature extraction with DINOv2 and/or VGG16 as well as classification with a random forest. It uses some utility and wrapper functions from conv_paint.

## Test script

The script [dino_tests.py](dino_tests.py) runs a (possibly large) set of tests and saves the results as a pandas data frame in a CSV file.

## Evaluation notebook

The notebook [dino_evaluate](dino_evaluate) provides interactive evaluation of outputs from the tests that can be run either through the notebook dino_paint.ipynb or the script dino_tests.py (each calling the function test_dino_forest() from the script dino_utils.py).
