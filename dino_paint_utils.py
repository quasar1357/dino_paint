import napari
from napari_convpaint.conv_paint_utils import (train_classifier,
                                               extract_annotated_pixels)
import numpy as np
from skimage.transform import resize
import torch
from torchvision.transforms import ToTensor

def extract_single_image_dinov2_features(image, dinov2_model='s'):
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
    return features

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

def scale_to_patch(image, crop_to_patch=True, scale=1, interpolation_order=0, patch_size=(14,14), print_shapes=False):
    '''
    Scales an image to a multiple of the patch size; optionally first crops it down to a multiple of the patch size.
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
        patched_image_shape = get_patched_image_shape(in_shape)        
        print(f"Patched image shape: {patched_image_shape[0]} x {patched_image_shape[1]} patches")
    return image_scaled

def dino_features_to_space(features, image_shape, interpolation_order=0, patch_size=(14,14)):
    '''
    Converts DINOv2 features to an image of a given shape.
    '''
    # Calculate number of features
    num_features = features.shape[1]
    # Calculate the shape of the patched image (i.e. how many patches fit in the image)
    patched_image_shape = get_patched_image_shape(image_shape)
    # Reshape linear patches into 2D
    feature_image = features.reshape(patched_image_shape[0], patched_image_shape[1], num_features)
    # Upsample to the size of the original image
    feature_image = resize(feature_image, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    # Swap axes to get features in the first dimension
    feature_image = feature_image.transpose(2,0,1)
    return feature_image

def get_patched_image_shape(image_shape, patch_size=(14,14)):
    '''
    Calculate the shape of the patched image (i.e. how many patches fit in the image)
    '''
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    else:
        patched_image_shape = (int(image_shape[0]/patch_size[0]), int(image_shape[1]/patch_size[1]))
    return patched_image_shape

### PUT EVERYTHING TOGETHER ###

def train_dino_forest(image, labels, crop_to_patch=True, scale=1, upscale_order=1, dinov2_model='s', append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a label image, and trains a random forest classifier on the DINOv2 features of the image.
    Returns the random forest, the image and labels used for training (both scaled to DINOv2's patch size) and the DINOv2 feature space.
    '''
    image_scaled = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    # NOTE: interpolation order must be 0 (nearest) for labels
    labels_scaled = scale_to_patch(labels, crop_to_patch, scale, interpolation_order=0)
    # Round to integers and convert to uint8 (labels must be integers)
    labels_scaled = np.round(labels_scaled).astype(np.uint8)
    features = extract_single_image_dinov2_features(image_scaled, dinov2_model)
    feature_space = dino_features_to_space(features, image_scaled.shape, interpolation_order=upscale_order)
    # Optionally append the image itself as feature
    if append_image_as_feature:
        image_as_feature = ensure_rgb(image_scaled)
        image_as_feature = image_as_feature.transpose(2,0,1)
        feature_space = np.concatenate((image_as_feature, feature_space), axis=0)
    # Extract annotated pixels and train random forest
    features_annot, targets = extract_annotated_pixels(feature_space, labels_scaled, full_annotation=False)
    random_forest = train_classifier(features_annot, targets)
    if show_napari:
        viewer = napari.Viewer()
        viewer.add_image(feature_space)
        viewer.add_image(image_scaled.astype(np.int32))
        viewer.add_labels(labels_scaled)
    return random_forest, image_scaled, labels_scaled, feature_space

def predict_dino_forest(image, random_forest, crop_to_patch=True, scale=1, upscale_order=1, dinov2_model='s', append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a trained random forest classifier, and predicts labels for the image.
    Returns the predicted labels, the image used for prediction (scaled to DINOv2's patch size) and its DINOv2 feature space.
    '''
    image_scaled = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    features = extract_single_image_dinov2_features(image_scaled, dinov2_model)
    feature_space = dino_features_to_space(features, image_scaled.shape, interpolation_order=upscale_order)
    # Optionally append the image itself as feature
    if append_image_as_feature:
        image_as_feature = ensure_rgb(image_scaled)
        image_as_feature = image_as_feature.transpose(2,0,1)
        feature_space = np.concatenate((image_as_feature, feature_space), axis=0)
    # Use the interpolated feature space (optionally appended with other features); reshape back to linear pixels for prediction
    features = feature_space.reshape(feature_space.shape[0], -1).transpose()
    predictions = random_forest.predict(features)
    # Reshape predictions to an image of the image that was scaled for DINOv2
    predicted_labels = predictions.reshape(image_scaled.shape[0], image_scaled.shape[1])
    if show_napari:
        viewer = napari.Viewer()
        viewer.add_image(feature_space)
        viewer.add_image(image_scaled.astype(np.int32))
        viewer.add_labels(predicted_labels)
    return predicted_labels, image_scaled, feature_space

def selfpredict_dino_forest(image, labels, crop_to_patch=True, scale=1, upscale_order=1, dinov2_model='s', append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a label image, and trains a random forest classifier on the DINOv2 features of the image.
    Then uses the trained random forest to predict labels for the image itself.
    Returns the predicted labels, the image and labels used for training (both scaled to DINOv2's patch size) and the DINOv2 feature space.
    '''
    train = train_dino_forest(image, labels, crop_to_patch, scale, upscale_order, dinov2_model, append_image_as_feature, show_napari=False)
    random_forest, image_scaled, labels_scaled, feature_space = train
    # Use the interpolated feature space (optionally appended with other features); reshape back to linear pixels for prediction
    features = feature_space.reshape(feature_space.shape[0], -1).transpose()
    # Directly use the already extracted features for prediction
    predictions = random_forest.predict(features)
    # Reshape predictions to an image of the image that was scaled for DINOv2
    predicted_labels = predictions.reshape(image_scaled.shape[0], image_scaled.shape[1])
    if show_napari:
        viewer = napari.Viewer()
        viewer.add_image(feature_space)    
        viewer.add_image(image_scaled.astype(np.int32))
        viewer.add_labels(labels_scaled)
        viewer.add_labels(predicted_labels)
    return predicted_labels, image_scaled, labels_scaled, feature_space