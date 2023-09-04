# dino_paint

dino_paint is an implementation of semantic segmentation using Meta AI's [DINOv2](https://github.com/facebookresearch/dinov2) for feature extraction and a random forest algorithm for classification. It is based on [conv_paint](https://github.com/guiwitz/napari-convpaint) by Guillaume Witz (original idea by Lucien Hinderling), which uses VGG16 for feature extraction instead. 

## Notebook dino_paint.ipynb

This is a first implementation using the notebook of conv_paint as a basis, replacing the feature extraction part with DINOv2. It still uses some utility and wrapper functions from conv_paint, which is, therefore, a requirement.