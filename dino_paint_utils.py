from napari_convpaint.conv_paint_utils import (train_classifier,
                                               predict_image,
                                               extract_annotated_pixels)
import numpy as np
from skimage.transform import resize
import torch
from torchvision.transforms import ToTensor

def extract_single_image_dinov2_features(image, labels, dinov2_model='s', crop_to_patch=True, scale=1, print_shapes=False):
    models = {'s': 'dinov2_vits14',
              'b': 'dinov2_vitb14',
              'l': 'dinov2_vitl14',
              'g': 'dinov2_vitg14'}
    model = torch.hub.load('facebookresearch/dinov2', models[dinov2_model], pretrained=True)
    dinov2_mean, dinov2_sd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    dinov2_patch_size = (14,14)

    image_scaled, labels_scaled = scale_to_patch(image, labels, crop_to_patch=crop_to_patch, scale=scale, patch_size=dinov2_patch_size, print_shapes=print_shapes)
    
    image_scaled = image_scaled.astype(np.float32)
    labels_scaled = labels_scaled.astype(np.int32)

    if image_scaled.ndim == 2:
        image_rgb = np.stack((image_scaled,)*3, axis=-1)
    else:
        image_rgb = image_scaled

    image_norm = normalize_np_array(image_rgb, dinov2_mean, dinov2_sd, axis = (0,1))
    image_tensor = ToTensor()(image_norm).float()

    features = extract_single_tensor_dinov2_features(image_tensor, model)

    return features, image_scaled, labels_scaled

def scale_to_patch(image, labels, crop_to_patch=True, scale=1, patch_size=(14,14), print_shapes=False):
    if crop_to_patch:
        crop_shape = (int(np.floor(image.shape[0]/patch_size[0]))*patch_size[0],
                    int(np.floor(image.shape[1]/patch_size[1]))*patch_size[1])
        in_shape = (crop_shape[0] * scale, crop_shape[1] * scale)

    else:
        crop_shape = image.shape
        in_shape = (int(np.ceil(image.shape[0]/patch_size[0]))*patch_size[0] * scale,
                    int(np.ceil(image.shape[1]/patch_size[1]))*patch_size[1] * scale)

    image_cropped = image[0:crop_shape[0], 0:crop_shape[1]]
    labels_cropped = labels[0:crop_shape[0], 0:crop_shape[1]]
    image_scaled = resize(image_cropped, in_shape, mode='edge', order=1, preserve_range=True)
    labels_scaled = resize(labels_cropped, in_shape, mode='edge', order=0, preserve_range=True)

    # Calculate the shape of the patched image (i.e. how many patches fit in the image)
    if not (in_shape[0]%patch_size[0] == 0 and in_shape[1]%patch_size[1] == 0):
        raise ValueError('Input shape must be divisible by patch size')
    else:
        patched_image_shape = (int(in_shape[0]/patch_size[0]), int(in_shape[1]/patch_size[1]))
    
    if print_shapes:
        print(f"Original image is: {image.shape[0]} x {image.shape[1]} pixels")
        print(f"Image is cropped to: {crop_shape[0]} x {crop_shape[1]} pixels")
        print(f"Shape of input used for model (multiple of patch size): {in_shape[0]} x {in_shape[1]} pixels")
        print(f"Patched image shape: {patched_image_shape[0]} x {patched_image_shape[1]} patches")

    return image_scaled, labels_scaled

def normalize_np_array(array, new_mean, new_sd, axis=(0,1)):
    current_mean, current_sd = np.mean(array, axis=axis), np.std(array, axis=axis)
    new_mean, new_sd = np.array(new_mean), np.array(new_sd)
    array_norm = (array - current_mean) / current_sd
    array_norm = array_norm * new_sd + new_mean
    
    return array_norm

def extract_single_tensor_dinov2_features(image_tensor, model):
    image_batch = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features_dict = model.forward_features(image_batch)
        features = features_dict['x_norm_patchtokens']
    
    # Convert to numpy array
    features = features.numpy()
    # Only take the image of interest out of the batch
    features = features[0,:,:]    

    return features

def dino_features_to_image(features, image_shape, patch_size=(14,14), pic_index=0, interpolation_order=0):
    # Calculate number of features
    num_features = features.shape[1]

    # Calculate the shape of the patched image (i.e. how many patches fit in the image)
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    else:
        patched_image_shape = (int(image_shape[0]/patch_size[0]), int(image_shape[1]/patch_size[1]))

    # Reshape linear patches into 2D
    feature_image = features.reshape(patched_image_shape[0], patched_image_shape[1], num_features)
    # Upsample to the size of the original image
    feature_image = resize(feature_image, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    # Swap axes to get features in the first dimension
    feature_image = feature_image.transpose(2,0,1)

    return feature_image

def predict_to_image(predictions, image_shape, patch_size=(14,14), interpolation_order=0):
    # Calculate the shape of the patched image (i.e. how many patches fit in the image)
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    else:
        patched_image_shape = (int(image_shape[0]/patch_size[0]), int(image_shape[1]/patch_size[1]))

    # Reshape linear patches into 2D
    predicted_image = predictions.reshape(patched_image_shape[0], patched_image_shape[1])
    # Upsample to the size of the original image
    predicted_image = resize(predicted_image, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    predicted_image = np.round(predicted_image).astype(np.uint8)

    return predicted_image