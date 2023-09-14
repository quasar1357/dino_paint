import napari
from napari_convpaint.conv_paint_utils import (train_classifier,
                                               extract_annotated_pixels)
import numpy as np
from skimage.transform import resize
import torch
from torchvision.transforms import ToTensor

### FEATURE EXTRACTION ###

# DINOv2 Feature Extraction

def extract_dinov2_features(image, upscale_order=1, dinov2_model='s'):
    '''
    Extracts features from a single image using a DINOv2 model. Returns a numpy array of shape (num_patches, num_features).
    '''
    models = {'s': 'dinov2_vits14',
              'b': 'dinov2_vitb14',
              'l': 'dinov2_vitl14',
              'g': 'dinov2_vitg14'}
    model = torch.hub.load('facebookresearch/dinov2', models[dinov2_model], pretrained=True)
    dinov2_mean, dinov2_sd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_rgb = ensure_rgb(image)
    image_norm = normalize_np_array(image_rgb, dinov2_mean, dinov2_sd, axis = (0,1))
    image_tensor = ToTensor()(image_norm).float()
    features = extract_single_tensor_dinov2_features(image_tensor, model)
    feature_space = dino_features_to_space(features, image.shape, interpolation_order=upscale_order)
    return feature_space

def extract_single_tensor_dinov2_features(image_tensor, model):
    '''
    Extracts features from a single image tensor using a DINOv2 model.
    '''
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0)
    # Extract features
    with torch.no_grad():
        features_dict = model.forward_features(image_batch)
        features = features_dict['x_norm_patchtokens']
    # Convert to numpy array
    features = features.numpy()
    # Remove batch dimension
    features = features[0,:,:]    
    return features

# VGG16 feature extraction (adjusted wrapper)

from napari_convpaint.conv_paint_utils import Hookmodel, get_features_current_layers

def extract_vgg16_features(image, layers, show_napari=False):
    '''
    Extracts features from a single image at given layers of the VGG16 model. Returns a numpy array of shape (num_features, W, H).
    '''
    image = ensure_rgb(image)
    model = Hookmodel(model_name='vgg16')
    all_layers = [key for key in model.module_dict.keys()]
    if (all(isinstance(layer, str) and layer in all_layers for layer in layers)):
        pass
    elif all(isinstance(layer, int) and 0 <= layer < len(all_layers) for layer in layers):
        layers = [all_layers[layer] for layer in layers]
    else:
        raise ValueError(f'The given layers are not valid. Please choose from the following layers (index as int or complete name as string): {all_layers}')
    # register hooks for the chosen layers in the model
    model.register_hooks(selected_layers=layers)
    # Get features for the whole image (pretending the whole image is annotated) without scaling
    fake_annot = np.ones(image.shape)
    features, targets = get_features_current_layers(
        model=model, image=image, annotations=fake_annot, scalings=[1], use_min_features=False, order=1)
    # Convert the DataFrame to a numpy array
    features_array = features.values
    # Reshape the features array to match the image shape
    feature_space = features_array.reshape(*image.shape[0:2], -1)
    feature_space = feature_space.transpose(2,0,1)
    # Now we can view the feature space using napari
    if show_napari:
        show_results_napari(image, feature_space)
    return feature_space

# Combine features

def extract_feature_space(image, upscale_order=1, dinov2_model='s', vgg16_layers=None, append_image_as_feature=False):
    '''
    Extracts features from the image given the DINOv2 model and/or VGG16 layers.
    '''
    # Check if dinov2_model and/or vgg16_layers are specified, and use one or both to create the feature space
    if dinov2_model is None and vgg16_layers is None:
        raise ValueError('Please specify a DINOv2 model and/or VGG16 layers to extract features from')
    elif vgg16_layers is None:
        feature_space = extract_dinov2_features(image, upscale_order, dinov2_model)
    elif dinov2_model is None:
        feature_space = extract_vgg16_features(image, vgg16_layers)
    else:
        dinov2_features = extract_dinov2_features(image, upscale_order, dinov2_model)
        vgg16_features = extract_vgg16_features(image, vgg16_layers)
        feature_space = np.concatenate((dinov2_features, vgg16_features), axis=0)
    # Optionally append the image itself as feature (3 channels, rgb)
    if append_image_as_feature:
        image_as_feature = ensure_rgb(image)
        image_as_feature = image_as_feature.transpose(2,0,1)
        feature_space = np.concatenate((image_as_feature, feature_space), axis=0)
    return feature_space

### PREDICTION ###

def predict_space_to_image(feature_space, random_forest):
    '''
    Predicts labels for a feature space using a trained random forest classifier, returning predictions in the same shape as the image/feature space.
    '''
    # linearize features (flatten spacial dimensions) for prediciton
    features = feature_space.reshape(feature_space.shape[0], -1).transpose()
    predictions = random_forest.predict(features)
    # Reshape predictions back to size of the image, which are the 2nd and 3rd dimensions of the feature space (1st = features)
    predicted_labels = predictions.reshape(feature_space.shape[1], feature_space.shape[2])
    return predicted_labels

### PUT EVERYTHING TOGETHER ###

def train_dino_forest(image, labels, crop_to_patch=True, scale=1, upscale_order=1, dinov2_model='s', vgg16_layers=None, append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a label image, and trains a random forest classifier on the DINOv2 features of the image.
    Returns the random forest, the image and labels used for training (both scaled to DINOv2's patch size) and the DINOv2 feature space.
    '''
    image_scaled = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    feature_space = extract_feature_space(image_scaled, upscale_order, dinov2_model, vgg16_layers, append_image_as_feature)
    # NOTE: interpolation order must be 0 (nearest) for labels
    labels_scaled = scale_to_patch(labels, crop_to_patch, scale, interpolation_order=0)
    # Round to integers and convert to uint8 (labels must be integers); this step should technically not be necessary with interpolation order = 0
    labels_scaled = np.round(labels_scaled).astype(np.uint8)
    # Extract annotated pixels and train random forest
    features_annot, targets = extract_annotated_pixels(feature_space, labels_scaled, full_annotation=False)
    random_forest = train_classifier(features_annot, targets)
    # Optionally show everything in Napari
    if show_napari:
        show_results_napari(image=image_scaled, feature_space=feature_space, labels=labels_scaled)
    return random_forest, image_scaled, labels_scaled, feature_space

def predict_dino_forest(image, random_forest, ground_truth=None, crop_to_patch=True, scale=1, upscale_order=1, dinov2_model='s', vgg16_layers=None, append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a trained random forest classifier, and predicts labels for the image.
    Returns the predicted labels, the image used for prediction (scaled to DINOv2's patch size) and its DINOv2 feature space.
    '''
    image_scaled = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    feature_space = extract_feature_space(image_scaled, upscale_order, dinov2_model, vgg16_layers, append_image_as_feature)
    # Use the interpolated feature space (optionally appended with other features) for prediction
    predicted_labels = predict_space_to_image(feature_space, random_forest)
    # Optionally show everything in Napari
    if show_napari:
        show_results_napari(image=image_scaled, feature_space=feature_space, predicted_labels=predicted_labels)
    # Optionally compare to ground truth and calculate accuracy    
    if ground_truth is not None:
        ground_truth_scaled = scale_to_patch(ground_truth, crop_to_patch, scale, interpolation_order=0)
        if not ground_truth_scaled.shape == predicted_labels.shape:
            raise ValueError('Ground truth and predicted labels must have the same shape')
        else:
            accuracy = np.sum(ground_truth_scaled == predicted_labels) / ground_truth_scaled.size
            # print(f'Accuracy of the prediction: {np.round(100*accuracy, 2)}%')
            return accuracy
    return predicted_labels, image_scaled, feature_space

def selfpredict_dino_forest(image, labels, crop_to_patch=True, scale=1, upscale_order=1, dinov2_model='s', vgg16_layers=None, append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a label image, and trains a random forest classifier on the DINOv2 features of the image.
    Then uses the trained random forest to predict labels for the image itself.
    Returns the predicted labels, the image and labels used for training (both scaled to DINOv2's patch size) and the DINOv2 feature space.
    '''
    train = train_dino_forest(image, labels, crop_to_patch, scale, upscale_order, dinov2_model, vgg16_layers, append_image_as_feature, show_napari=False)
    random_forest, image_scaled, labels_scaled, feature_space = train
    predicted_labels = predict_space_to_image(feature_space, random_forest)
    # Optionally show everything in Napari
    if show_napari:
        show_results_napari(image_scaled, feature_space, labels_scaled, predicted_labels)
    return predicted_labels, image_scaled, labels_scaled, feature_space

### HELPER FUNCTIONS ###

def ensure_rgb(image):
    '''
    Checks if an image is RGB, and if not, converts it to RGB.
    '''
    if image.ndim == 2:
        image_rgb = np.stack((image,)*3, axis=-1)
    else:
        image_rgb = image
    return image_rgb

def normalize_np_array(array, new_mean, new_sd, axis=(0,1)):
    '''
    Normalizes a numpy array to a new mean and standard deviation.
    '''
    current_mean, current_sd = np.mean(array, axis=axis), np.std(array, axis=axis)
    new_mean, new_sd = np.array(new_mean), np.array(new_sd)
    array_norm = (array - current_mean) / current_sd
    array_norm = array_norm * new_sd + new_mean
    return array_norm

def dino_features_to_space(features, image_shape, interpolation_order=1, patch_size=(14,14)):
    '''
    Converts DINOv2 features to an "image" of a given shape (= feature space).
    '''
    # Calculate number of features
    num_features = features.shape[1]
    # Calculate the shape of the patched image (i.e. how many patches fit in the image)
    patched_image_shape = get_patched_image_shape(image_shape, patch_size)
    # Reshape linear patches into 2D
    feature_space = features.reshape(patched_image_shape[0], patched_image_shape[1], num_features)
    # Upsample to the size of the original image
    feature_space = resize(feature_space, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    # Swap axes to get features in the first dimension
    feature_space = feature_space.transpose(2,0,1)
    return feature_space

def scale_to_patch(image, crop_to_patch=True, scale=1, interpolation_order=1, patch_size=(14,14), print_shapes=False):
    '''
    Scales an image to a multiple of the patch size; optionally first/instead crops it down to a multiple of the patch size.
    '''
    if crop_to_patch:
        crop_shape = (int(np.floor(image.shape[0]/patch_size[0]))*patch_size[0],
                    int(np.floor(image.shape[1]/patch_size[1]))*patch_size[1])
        in_shape = (crop_shape[0] * scale, crop_shape[1] * scale)
    else:
        crop_shape = image.shape
        in_shape = (int(np.ceil(image.shape[0]/patch_size[0]))*patch_size[0] * scale,
                    int(np.ceil(image.shape[1]/patch_size[1]))*patch_size[1] * scale)
    image_cropped = image[0:crop_shape[0], 0:crop_shape[1]]
    image_scaled = resize(image_cropped, in_shape, mode='edge', order=interpolation_order, preserve_range=True)
    # Optionally print all the shapes to check if everything is correct
    if print_shapes:
        print(f"Original image is: {image.shape[0]} x {image.shape[1]} pixels")
        print(f"Image is cropped to: {crop_shape[0]} x {crop_shape[1]} pixels")
        print(f"Shape of input used for model (multiple of patch size): {in_shape[0]} x {in_shape[1]} pixels")
        # Calculate the shape of the patched image (i.e. how many patches fit in the image)
        patched_image_shape = get_patched_image_shape(in_shape, patch_size)        
        print(f"Patched image shape: {patched_image_shape[0]} x {patched_image_shape[1]} patches")
    return image_scaled

def get_patched_image_shape(image_shape, patch_size=(14,14)):
    '''
    Calculates the shape of the patched image (i.e. how many patches fit in the image)
    '''
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    else:
        patched_image_shape = (int(image_shape[0]/patch_size[0]), int(image_shape[1]/patch_size[1]))
    return patched_image_shape

def show_results_napari(image=None, feature_space=None, labels=None, predicted_labels=None):
    '''
    Shows the results of a DINOv2 feature extraction and/or prediction together with the image in napari.
    '''
    viewer = napari.Viewer()
    if feature_space is not None: viewer.add_image(feature_space)
    if image is not None: viewer.add_image(image.astype(np.int32))
    if labels is not None: viewer.add_labels(labels)
    if predicted_labels is not None: viewer.add_labels(predicted_labels)
    return viewer
