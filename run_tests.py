
import numpy as np
import skimage
from dino_paint_utils import test_dino_forest
import PIL
import urllib
import datetime

def load_url_to_array(url, from_github=False):
    return np.array(PIL.Image.open(urllib.request.urlopen(['', 'https://github.com/quasar1357/dino_paint/raw/main/images_and_labels/'][from_github] + url)))

# Load/define image
image_to_train = skimage.data.astronaut()#[200:300,250:400]
labels_to_train = load_url_to_array('astro_labels.tif', True)#[200:300,250:400]
image_to_pred = None #skimage.data.camera()
ground_truth = load_url_to_array('astro_ground_truth.tif', True)#[200:300,250:400]

# Create possible combinations of VGG16 layers and scales
all_vggs = [0,2,5,7,10,12,14,17,19,21,24,26,28]
chosen_vggs = [0,2,5,7,10,12,14,17,19,21,24,26,28]
single_vggs = [[i] for i in chosen_vggs]
consecutive_vggs = [chosen_vggs[:s] for s in range(1, 1 + len(chosen_vggs))]
dual_vggs = [[chosen_vggs[i], chosen_vggs[j]] for i in range(len(chosen_vggs)) for j in range(i+1, len(chosen_vggs))]

all_scales = [1.0, 1.32, 1.73, 2.28, 3.0]
chosen_scales = all_scales[:3]
single_scales = [{"DINOv2": [d], "VGG16": [v]} for d in chosen_scales for v in chosen_scales]
consecutive_scales = [{"DINOv2": chosen_scales[:d], "VGG16": chosen_scales[:v]} for d in range(1, 1 + len(chosen_scales)) for v in range(1, 1 + len(chosen_scales))]

# Choose what to use
dino_models = [None, 's']#, 'b']
dino_layer_combos = [(),
                     [8, 9, 10, 11]]
vgg_layer_combos = [None,
                    [2, 7]]#,
                    # [24,26,28]]
scale_combos = [{"DINOv2": (), "VGG16": ()},
                {"DINOv2": (), "VGG16": (1,2)}]

# Run test
test = test_dino_forest(image_to_train, labels_to_train, ground_truth, image_to_pred,
                        dinov2_models=dino_models, dinov2_layer_combos=dino_layer_combos, vgg16_layer_combos=vgg_layer_combos, scale_combos=scale_combos,
                        print_avg=True, print_best=False, write_files=True)
