import napari
from napari_convpaint.conv_paint_utils import (train_classifier,
                                               extract_annotated_pixels)
import numpy as np
from skimage.transform import resize
import torch
from torchvision.transforms import ToTensor

def extract_single_image_dinov2_features(image, dinov2_model='s'):
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
    
    if print_shapes:
        print(f"Original image is: {image.shape[0]} x {image.shape[1]} pixels")
        print(f"Image is cropped to: {crop_shape[0]} x {crop_shape[1]} pixels")
        print(f"Shape of input used for model (multiple of patch size): {in_shape[0]} x {in_shape[1]} pixels")
        # Calculate the shape of the patched image (i.e. how many patches fit in the image)
        patched_image_shape = get_patched_image_shape(in_shape)        
        print(f"Patched image shape: {patched_image_shape[0]} x {patched_image_shape[1]} patches")

    return image_scaled

def dino_features_to_image(features, image_shape, interpolation_order=0, patch_size=(14,14)):
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

def predict_to_image(predictions, image_shape, interpolation_order=0, patch_size=(14,14)):
    '''
    Converts DINOv2 predictions to an image of a given shape.
    '''
    # Calculate the shape of the patched image (i.e. how many patches fit in the image)
    patched_image_shape = get_patched_image_shape(image_shape)
    # Reshape linear patches into 2D
    predicted_image = predictions.reshape(patched_image_shape[0], patched_image_shape[1])
    # Upsample to the size of the original image
    predicted_image = resize(predicted_image, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    # Round to integers and convert to uint8 (labels must be integers)
    predicted_image = np.round(predicted_image).astype(np.uint8)
    return predicted_image

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

def train_dino_forest(image, labels, crop_to_patch=True, scale=1, dinov2_model='s', show_napari=False):
    image_to_train = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    labels_to_train = scale_to_patch(labels, crop_to_patch, scale, interpolation_order=0)
    features_train = extract_single_image_dinov2_features(image_to_train, dinov2_model)
    features_space_train = dino_features_to_image(features_train, image_to_train.shape)
    features_annot, targets = extract_annotated_pixels(features_space_train, labels_to_train, full_annotation=False)
    random_forest = train_classifier(features_annot, targets)
    if show_napari:
        viewer = napari.Viewer()
        viewer.add_image(image_to_train.astype(np.int32))
        viewer.add_labels(labels_to_train)
        viewer.add_image(features_space_train)
    return random_forest, image_to_train, labels_to_train, features_space_train

def predict_dino_forest(image, random_forest, crop_to_patch=True, scale=1, dinov2_model='s', show_napari=False):
    image_to_predict = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    features_predict = extract_single_image_dinov2_features(image_to_predict, dinov2_model)
    features_space_predict = dino_features_to_image(features_predict, image_to_predict.shape)
    predictions = random_forest.predict(features_predict)
    predicted_labels = predict_to_image(predictions, image_to_predict.shape, interpolation_order=0)
    if show_napari:
        viewer = napari.Viewer()
        viewer.add_image(image_to_predict.astype(np.int32))
        viewer.add_labels(predicted_labels)
        viewer.add_image(features_space_predict)
    return predicted_labels, image_to_predict, features_space_predict

def selfpredict_dino_forest(image, labels, crop_to_patch=True, scale=1, dinov2_model='s'):
    image_to_train = scale_to_patch(image, crop_to_patch, scale, interpolation_order=1)
    labels_to_train = scale_to_patch(labels, crop_to_patch, scale, interpolation_order=0)
    features_train = extract_single_image_dinov2_features(image_to_train, dinov2_model)
    features_space_train = dino_features_to_image(features_train, image_to_train.shape)
    features_annot, targets = extract_annotated_pixels(features_space_train, labels_to_train, full_annotation=False)
    random_forest = train_classifier(features_annot, targets)
    # Directly use the features already extracted for prediction
    image_to_predict = image_to_train
    features_predict = features_train
    features_space_predict = features_space_train
    predictions = random_forest.predict(features_predict)
    predicted_labels = predict_to_image(predictions, image_to_predict.shape, interpolation_order=0)
    return predicted_labels, image_to_predict, features_space_predict