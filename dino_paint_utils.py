import napari
from napari_convpaint.conv_paint_utils import (train_classifier,
                                               extract_annotated_pixels,
                                               Hookmodel,
                                               get_features_current_layers)
import numpy as np
from skimage.transform import resize
import torch
from torchvision.transforms import ToTensor

### FEATURE EXTRACTION ###

# Store loaded DINOv2 models in a global dictionary to avoid loading the same model multiple times
loaded_dinov2_models = {}

# DINOv2 Feature Extraction

def extract_single_tensor_dinov2_features(image_tensor, model, layers=()):
    '''
    Extracts features from a single image tensor using a DINOv2 model.
    '''
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0)
    # Extract features
    if layers:
        # print(f"Using DINOv2 layers {layers}")
        with torch.no_grad():
            features = model.get_intermediate_layers(image_batch, n = layers, reshape=False)
        # Convert to numpy array
        features = features.numpy()
        # Concatenate the channels of the intermediate layers (initially split in the first dimension) in the last dimension
        num_layers, num_batches, num_patches, num_channels = features.shape
        features = np.transpose(features, (1, 2, 3, 0))
        features = np.reshape(features, (num_batches, num_patches, num_channels * num_layers))
    else:
        # print("using DINOv2 x_norm_patchtokens")
        with torch.no_grad():
            features_dict = model.forward_features(image_batch)
            features = features_dict['x_norm_patchtokens']
        # Convert to numpy array
        features = features.numpy()
    # Remove batch dimension
    features = features[0]
    return features

def extract_dinov2_features(image, dinov2_model='s', layers=(), upscale_order=0):
    '''
    Extracts features from a single image using a DINOv2 model. Returns a numpy array of shape (num_patches, num_features).
    '''
    models = {'s': 'dinov2_vits14',
              'b': 'dinov2_vitb14',
              'l': 'dinov2_vitl14',
              'g': 'dinov2_vitg14',
              's_r': 'dinov2_vits14_reg',
              'b_r': 'dinov2_vitb14_reg',
              'l_r': 'dinov2_vitl14_reg',
              'g_r': 'dinov2_vitg14_reg'}
    dinov2_mean, dinov2_sd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    dinov2_name = models[dinov2_model]
    if dinov2_name not in loaded_dinov2_models:
        # print(f"Loading DINOv2 model {dinov2_name}")
        loaded_dinov2_models[dinov2_name] = torch.hub.load('facebookresearch/dinov2', dinov2_name, pretrained=True, verbose=False)
    model = loaded_dinov2_models[dinov2_name]
    model.eval()
    image_rgb = ensure_rgb(image)
    image_norm = normalize_np_array(image_rgb, dinov2_mean, dinov2_sd, axis = (0,1))
    image_tensor = ToTensor()(image_norm).float()
    features = extract_single_tensor_dinov2_features(image_tensor, model, layers)
    feature_space = dino_features_to_space(features, image.shape, interpolation_order=upscale_order)
    return feature_space

def pad_and_extract_dinov2_features(image, dinov2_model='s', layers=(), upscale_order=0, pad_mode='reflect', extra_pads=()):
    '''
    Pad an image to the next multiple of patch size, extract DINOv2 features
    Optionally use (several) extra paddings in order to shift the patch position
    '''
    # Pad the scaled image to patch size
    image_padded = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, extra_pad=False, patch_size=(14,14))
    # Extract features using the scaled and padded image
    dinov2_features = extract_dinov2_features(image_padded, dinov2_model, layers, upscale_order)
    # Crop the features back to original size of the scaled image
    dinov2_features = dinov2_features[:,:image.shape[0], :image.shape[1]]
    # If any extra pads are given, extract DINOv2 on image with those extra_pads added to the left and concatenate features
    for extra_pad in extra_pads:
        # Pad the image to patch size using the extra padding provided
        image_extra_padded = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, extra_pad=extra_pad, patch_size=(14,14))
        # Extract features using the scaled and padded image
        features_extra_padded = extract_dinov2_features(image_extra_padded, dinov2_model, layers, upscale_order)
        # Crop the features back to original size of the scaled image (removing the extra padding added)
        features_extra_padded = features_extra_padded[:, extra_pad:extra_pad+image.shape[0], extra_pad:extra_pad+image.shape[1]]
        # Add the features to the features extracted so far
        dinov2_features = np.concatenate((dinov2_features, features_extra_padded), axis=0)
    return dinov2_features

# VGG16 feature extraction (adjusted wrapper)

def extract_vgg16_features(image, layers, show_napari=False):
    '''
    Extracts features from a single image at given layers of the VGG16 model. Returns a numpy array of shape (num_features, W, H).
    '''
    model = Hookmodel(model_name='vgg16')
    all_layers = [key for key in model.module_dict.keys()]
    # If the layers are given as strings, and they are valid, we can use them directly
    if (all(isinstance(layer, str) and layer in all_layers for layer in layers)):
        pass
    # If the layers are given as ints, we need to extract the strings first
    elif all(isinstance(layer, int) and 0 <= layer < len(all_layers) for layer in layers):
        layers = [all_layers[layer] for layer in layers]
    # If the layers are not valid, raise an error
    else:
        raise ValueError(f'The given layers are not valid. Please choose from the following layers (index as int or complete name as string): {all_layers}')
    # Register hooks for the chosen layers in the model
    # print(f"Using VGG16 layers {layers}")
    model.register_hooks(selected_layers=layers)
    # Since so far, conv_paint only handles movies of 2D (= grey-scale) images, we take the mean of the 3 channels if the image is RGB
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    # Create fake annotations (pretending the whole image is annotated), because conv_paint only returns features for annotated pixels    
    fake_annot = np.ones(image.shape)
    # Get features and targets using the fake annotations covering the full image
    features, targets = get_features_current_layers(
        model=model, image=image, annotations=fake_annot, scalings=(1,), use_min_features=False, order=1)
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

def extract_feature_space_multi_scale(image, extraction_func, scales=(), **kwargs):
    '''
    Extract features with (possibly) different scalings, using a given extraction function
    '''
    # If no scales are given, just extract with unscaled image
    if not scales:
        return extraction_func(image, **kwargs)
    # Sample the features extracted for all different scales in a list
    features_list = []
    for scale in scales:
        # If scale factor is 1, skip the scaling, since it is not necessary and uses ressources
        if scale == 1:
            image_scaled = image
        # Scale the image with the scale factor given
        else:
            image_scaled = resize(image, (image.shape[0] * scale, image.shape[1] * scale), mode='edge', order=0, preserve_range=True)
        # Extract the features from the scaled image
        features_scaled = extraction_func(image_scaled, **kwargs)
        # If scale factor is 1, skip the rescaling, since it is not necessary and uses ressources
        if scale == 1:
            features_rescaled = features_scaled
        # Scale the features back down to image size
        else:
            num_features = features_scaled.shape[0]
            features_rescaled = resize(features_scaled, (num_features, image.shape[0], image.shape[1]), mode='edge', order=0, preserve_range=True)
        # Append the features to the features_list
        features_list.append(features_rescaled)
    # Concatenate all features from different scalings
    features = np.concatenate(features_list, axis=0)
    return features

def extract_feature_space(image, dinov2_model='s', dinov2_layers=(), upscale_order=0, pad_mode='reflect', extra_pads=(), vgg16_layers=None, append_image_as_feature=False, scales=()):
    '''
    Extracts features from the image given the DINOv2 model and/or VGG16 layers. Applies padding for DINOv2 to conform with patch size.
    Additional features with extra padding (option extra_pads) can be concatenated to shift patch pattern (and increase resolution).
    '''
    features_list = []
    # Extract features with DINOv2
    if dinov2_model is not None:
        dinov2_scales = scales["DINOv2"] if scales else scales
        dinov2_features = extract_feature_space_multi_scale(image, pad_and_extract_dinov2_features, dinov2_scales, dinov2_model=dinov2_model, layers=dinov2_layers, upscale_order=upscale_order, pad_mode=pad_mode, extra_pads=extra_pads)
        features_list.append(dinov2_features)

    # Extract features with VGG16
    if vgg16_layers is not None:
        vgg16_scales = scales["VGG16"] if scales else scales
        vgg16_features = extract_feature_space_multi_scale(image, extract_vgg16_features, vgg16_scales, layers=vgg16_layers)
        features_list.append(vgg16_features)

    # Optionally use the image itself as feature (3 channels, rgb)
    if append_image_as_feature:
        image_as_feature = ensure_rgb(image)
        image_as_feature = image_as_feature.transpose(2,0,1)
        features_list.append(image_as_feature)

    # Use the combination/choice of features given
    if dinov2_model is None and vgg16_layers is None and image_as_feature is None:
        raise ValueError('Please specify a DINOv2 model and/or VGG16 layers to extract features from')
    else:
        feature_space = np.concatenate(features_list, axis=0)
        
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

def train_dino_forest(image, labels, dinov2_model='s', dinov2_layers=(), upscale_order=0, pad_mode='reflect', extra_pads=(), vgg16_layers=None, append_image_as_feature=False, scales=(), show_napari=False):
    '''
    Takes an image and a label image, and trains a random forest classifier on the DINOv2 features of the image.
    Returns the random forest, the image and labels used for training (both scaled to DINOv2's patch size) and the DINOv2 feature space.
    '''
    feature_space = extract_feature_space(image, dinov2_model, dinov2_layers, upscale_order, pad_mode, extra_pads, vgg16_layers, append_image_as_feature, scales)
    # Extract annotated pixels and train random forest
    features_annot, targets = extract_annotated_pixels(feature_space, labels, full_annotation=False)
    random_forest = train_classifier(features_annot, targets)
    # Optionally show everything in Napari
    if show_napari:
        show_results_napari(image=image, feature_space=feature_space, labels=labels)
    return random_forest, image, labels, feature_space

def predict_dino_forest(image, random_forest, ground_truth=None, dinov2_model='s', dinov2_layers=(), upscale_order=0, pad_mode='reflect', extra_pads=(), scales=(), vgg16_layers=None, append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a trained random forest classifier, and predicts labels for the image.
    Returns the predicted labels, the image used for prediction (scaled to DINOv2's patch size) and its DINOv2 feature space.
    '''
    feature_space = extract_feature_space(image, dinov2_model, dinov2_layers, upscale_order, pad_mode, extra_pads, vgg16_layers, append_image_as_feature, scales)
    # Use the interpolated feature space (optionally appended with other features) for prediction
    predicted_labels = predict_space_to_image(feature_space, random_forest)
    # Optionally show everything in Napari
    if show_napari:
        show_results_napari(image=image, feature_space=feature_space, predicted_labels=predicted_labels, ground_truth=ground_truth)
    # Optionally compare to ground truth and calculate accuracy    
    accuracy = None
    if ground_truth is not None:
        if not ground_truth.shape == predicted_labels.shape:
            raise ValueError('Ground truth and predicted labels must have the same shape')
        else:
            accuracy = np.sum(ground_truth == predicted_labels) / ground_truth.size    
    return predicted_labels, image, feature_space, accuracy

def selfpredict_dino_forest(image, labels, ground_truth=None, dinov2_model='s', dinov2_layers=(), upscale_order=0, pad_mode='reflect', extra_pads=(), scales=(), vgg16_layers=None, append_image_as_feature=False, show_napari=False):
    '''
    Takes an image and a label image, and trains a random forest classifier on the DINOv2 features of the image.
    Then uses the trained random forest to predict labels for the image itself.
    Returns the predicted labels, the image and labels used for training (both scaled to DINOv2's patch size) and the DINOv2 feature space.
    '''
    train = train_dino_forest(image, labels, dinov2_model, dinov2_layers, upscale_order, pad_mode, extra_pads, vgg16_layers, append_image_as_feature, scales, show_napari=False)
    random_forest, image, labels, feature_space = train
    predicted_labels = predict_space_to_image(feature_space, random_forest)
    # Optionally show everything in Napari
    if show_napari:
        show_results_napari(image=image, feature_space=feature_space, labels=labels, predicted_labels=predicted_labels, ground_truth=ground_truth)
    # Optionally compare to ground truth and calculate accuracy
    accuracy = None 
    if ground_truth is not None:
        if not ground_truth.shape == predicted_labels.shape:
            raise ValueError('Ground truth and predicted labels must have the same shape')
        else:
            accuracy = np.sum(ground_truth == predicted_labels) / ground_truth.size
    return predicted_labels, image, labels, feature_space, accuracy

### HELPER FUNCTIONS ###

def ensure_rgb(image):
    '''
    Checks if an image is RGB, and if not, converts it to RGB.
    '''
    if image.ndim == 2:
        # image_rgb = (np.stack((image,)*3, axis=-1)/np.max(image)*255).astype(np.uint8)
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

def dino_features_to_space(features, image_shape, interpolation_order=0, patch_size=(14,14)):
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

def pad_to_patch(image, vert_pos="center", hor_pos="center", pad_mode='constant', extra_pad=0, patch_size=(14,14)):
    '''
    Pads an image to the next multiple of patch size.
    The pad position can be chosen on both axis in the tuple (vert, hor), where vert can be "top", "center" or "bottom" and hor can be "left", "center" or "right".
    Optionally add extra padding of total one patch size, distributed half/half on each side (shifts patch positions by half path size).
    pad_mode can be chosen according to numpy pad method.
    '''
    # If image is an rgb image, run this function on each channel
    if len(image.shape) == 3:
        channel_list = np.array([pad_to_patch(image[:,:, channel], vert_pos, hor_pos, pad_mode, extra_pad, patch_size) for channel in range(image.shape[2])])
        rgb_padded = np.moveaxis(channel_list, 0, 2)
        return rgb_padded
    # For a greyscale image (or each separate RGB channel):
    h, w = image.shape
    # The height and width to consider for the padding to patch size are the image h and w plus the extra padding on each side
    h, w = h + 2 * extra_pad, w + 2 * extra_pad
    ph, pw = patch_size
    # Calculate how much padding has to be done in total on each axis
    # The total pad on one axis is a patch size minus whatever remains when dividing the picture size including the extra pads by the patch size
    # The  * (h % ph != 0) term (and same with wdith) ensure that the pad is 0 if the shape is already a multiple of the patch size
    vertical_pad = (ph - h % ph) * (h % ph != 0)
    horizontal_pad = (pw - w % pw) * (w % pw != 0)
    # Define the paddings on each side depending on the chosen positions
    top_pad = {"top": vertical_pad,
               "center": np.ceil(vertical_pad/2),
               "bottom": 0
               }[vert_pos]
    bot_pad = vertical_pad - top_pad
    left_pad = {"left": horizontal_pad,
                "center": np.ceil(horizontal_pad/2),
                "right": 0
                }[hor_pos]
    right_pad = horizontal_pad - left_pad
    # Add extra padding if given (option extra_pad)
    top_pad = top_pad + extra_pad
    bot_pad = bot_pad + extra_pad
    left_pad = left_pad + extra_pad
    right_pad = right_pad + extra_pad
    # Make sure paddings are ints
    top_pad, bot_pad, left_pad, right_pad = int(top_pad), int(bot_pad), int(left_pad), int(right_pad)
    # Pad the image using the pad sizes as calculated and the mode given as input
    image_padded = np.pad(image, ((top_pad, bot_pad), (left_pad, right_pad)), mode=pad_mode)
    return image_padded

def get_patched_image_shape(image_shape, patch_size=(14,14)):
    '''
    Calculates the shape of the patched image (i.e. how many patches fit in the image)
    '''
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    else:
        patched_image_shape = (int(image_shape[0]/patch_size[0]), int(image_shape[1]/patch_size[1]))
    return patched_image_shape

def show_results_napari(image=None, feature_space=None, labels=None, predicted_labels=None, ground_truth=None):
    '''
    Shows the results of a DINOv2 feature extraction and/or prediction together with the image in napari.
    '''
    viewer = napari.Viewer()
    if feature_space is not None: viewer.add_image(feature_space)
    if image is not None: viewer.add_image(image.astype(np.int32))
    if labels is not None: viewer.add_labels(labels)
    if predicted_labels is not None: viewer.add_labels(predicted_labels)
    if ground_truth is not None: viewer.add_labels(ground_truth)
    return viewer

### TESTS USING GROUND TRUTH ###

def test_dino_forest(image_to_train, labels_to_train, ground_truth, image_to_pred=None, dinov2_models=('s',), dinov2_layer_combos=((),), vgg16_layer_combos=(None,), scale_combos=((),), print_avg=False, print_max=False):
    '''
    Tests prediction accuracy of a DINOv2 model and/or VGG16 model trained on a given image against a ground truth
    Tests for different scales, DINOv2 models, DINOv2 layers, VGG16 layers and "image as feature" options.
    Tests prediction on a given image, or, if no image to predict is specified (=None), on the image used for training itself (selfprediction).
    '''
    from time import time
    # Some options are held constant and not looped over; they are defined here
    upscale_order=0
    pad_mode='reflect'
    extra_pads=()
    im_feat=True
    # Prepare matrix for results
    total_dinov2_combos = len(dinov2_models) * len(dinov2_layer_combos)
    acc_shape = (total_dinov2_combos, len(vgg16_layer_combos), len(scale_combos))
    # Prepare matrix for accuracies with all parameter combinations; use 0.5 as default, since this is the expectation of accuracy with 2 labels
    accuracies = np.full(acc_shape, 0.0)
    ex_times = np.full(acc_shape, 0.0)
    # Loop over all possible combinations
    for d_m_i, dino in enumerate(dinov2_models):
        for d_l_i, d_layers in enumerate(dinov2_layer_combos):
            print(f'Running tests for DINOv2 model {dino}, layers {d_layers}...')
            # Linearize the combinations of DINOv2 models and layer selection
            d_i = len(dinov2_layer_combos) * d_m_i + d_l_i
            for v_i, vgg in enumerate(vgg16_layer_combos):
                print(f'    Running tests for VGG16 layers {vgg}...')
                for s_i, s in enumerate(scale_combos):
                    print(f'        Running tests with scale combination {s}...')
                    # get starting time
                    start = time()
                    # Selfpredict if no image to predict is specified
                    if image_to_pred is None:
                        pred = selfpredict_dino_forest(image_to_train, labels_to_train, ground_truth, dinov2_model=dino, dinov2_layers=d_layers, upscale_order=upscale_order, pad_mode=pad_mode, extra_pads=extra_pads, scales=s, vgg16_layers=vgg, append_image_as_feature=im_feat, show_napari=False)
                    # Otherwise predict labels for the given image
                    else:
                        train = train_dino_forest(image_to_train, labels_to_train, dinov2_model=dino, dinov2_layers=d_layers, upscale_order=upscale_order, pad_mode=pad_mode, extra_pads=extra_pads, vgg16_layers=vgg, append_image_as_feature=im_feat, scales=s, show_napari=False)
                        random_forest = train[0]
                        pred = predict_dino_forest(image_to_pred, random_forest, ground_truth, dinov2_model=dino, dinov2_layers=d_layers, upscale_order=upscale_order, pad_mode=pad_mode, extra_pads=extra_pads, scales=s, vgg16_layers=vgg, append_image_as_feature=im_feat, show_napari=False)
                    # Save the accuracies in the matrix
                    accuracies[d_i, v_i, s_i] = pred[-1]
                    # Execution time is start time - current time
                    ex_times[d_i, v_i, s_i] = time() - start
    
    # Calculate averages and optionally print them
    avg_dinos = np.zeros(total_dinov2_combos)
    avg_vggs = np.zeros(len(vgg16_layer_combos))
    avg_scales = np.zeros(len(scale_combos))
    if print_avg:
        print("\n--- AVERAGES ---")
    for d_m_i, dino in enumerate(dinov2_models):
        for d_l_i, d_layers in enumerate(dinov2_layer_combos):   
            d_i = len(dinov2_layer_combos) * d_m_i + d_l_i
            avg_dino = np.mean(accuracies[d_i,:,:][accuracies[d_i,:,:]!=0])
            avg_dinos[d_i] = avg_dino
            if print_avg:
                print(f'Average accuracy for DINOv2 model {dino}, layers {d_layers}: {np.round(100*avg_dino, 2)}%')
    for v_i, vgg in enumerate(vgg16_layer_combos):
        avg_vgg = np.mean(accuracies[:,v_i,:][accuracies[:,v_i,:]!=0])
        avg_vggs[v_i] = avg_vgg
        if print_avg:
            print(f'Average accuracy for VGG16 layers {vgg}: {np.round(100*avg_vgg, 2)}%')
    for s_i, s in enumerate(scale_combos):
        avg_scale = np.mean(accuracies[:,:,s_i][accuracies[:,:,s_i]!=0])
        avg_scales[s_i] = avg_scale
        if print_avg:
            print(f'Average accuracy for scale {s}: {np.round(100*avg_scale, 2)}%')
    
    # Calculate the maximum accuracy and the corresponding parameters; optionally print them
    max_acc_idx = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    max_acc = accuracies[max_acc_idx]
    if print_max:
        best_dino_model = max_acc_idx[0] // len(dinov2_layer_combos)
        best_dino_layers = max_acc_idx[0] % len(dinov2_layer_combos)
        print("\n--- MAXIMUM ---\n"+
              f"The maximum accuracy {np.round(100*max_acc, 2)}% is reached with:\n"+
              f"    dino model = {dinov2_models[best_dino_model]}\n"+
              f"    dino layers = {dinov2_layer_combos[best_dino_layers]}\n"+
              f"    vgg16 = {vgg16_layer_combos[max_acc_idx[1]]}\n"+
              f"    scale = {scale_combos[max_acc_idx[2]]}")
    
    # Return the specific accuracies, the averages and the maximum
    return accuracies, (avg_dinos, avg_vggs, avg_scales), (max_acc, max_acc_idx), ex_times