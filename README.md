# dino_paint

dino_paint is an implementation of semantic segmentation using Meta AI's [DINOv2](https://github.com/facebookresearch/dinov2) for feature extraction and a random forest algorithm for classification. It is based on [conv_paint](https://github.com/guiwitz/napari-convpaint) by Guillaume Witz (original idea by Lucien Hinderling), which uses VGG16 for feature extraction instead. The latest version of dino_paint allows combination of any DINOv2 model with any selection of layers of the VGG16 network.

## Notebook dino_paint.ipynb

This notebook guides through standard applications such as feature extraction, training and prediciton. It also includes self-prediction, where features of an image are only extracted once, and used for training and prediction on itself. Finally, it enables running automated tests of accuracy and execution times using different model selections.

Note that the code still uses some utility and wrapper functions from conv_paint, which is, therefore, a requirement.

## dino_paint_utils.py

This script contains all the functions used for feature extraction with DINOv2 and/or VGG16 as well as classification with a random forest. It uses some utility and wrapper functions from conv_paint.