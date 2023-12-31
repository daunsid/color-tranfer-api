from typing import List
from typing import Dict
import itertools
import torch
import cv2
import os
import numpy
from google.colab.patches import cv2_imshow

import pathlib
import functools

import torch
from torchvision import transforms
from google.colab.patches import cv2_imshow

from color_transfer.transfer_gen.segment_net.bisenetv2 import BiSeNetV2
from color_transfer.transfer_gen.segment_net.visualisation import draw_results
import matplotlib.pyplot as plt

from PIL import Image as im
import cv2
import numpy as np
import skimage.exposure



def load_image(image_path):
    image = cv2.imread(image_path)
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


modelweight="data/model_segmentation_realtime_skin_30.pth"
state_dict = torch.load(modelweight,map_location=torch.device('cpu'))
model = BiSeNetV2(['skin'])
model.load_state_dict(state_dict)
model.eval()


def createSkinMask(targetimage):
    
    targetimage = cv2.cvtColor(targetimage, cv2.COLOR_BGR2RGB)

    image_width = (targetimage.shape[1] // 32) * 32
    image_height = (targetimage.shape[0] // 32) * 32

    resized_image = targetimage[:image_height, :image_width]

    fn_image_transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])

    transformed_image = fn_image_transform(resized_image)

    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0)
        results = model(transformed_image)['out']
        results = torch.sigmoid(results)
        
        results = results > 0.5
        mask=results[0]
        mask=mask.squeeze(0)
        mask = mask.cpu().numpy()
        mask=mask*255
        mask = mask.astype('uint8')

    return mask,resized_image

def create_segmentaion(im_path):
    original = cv2.imread(im_path)

    mask, resized_image = createSkinMask(original)


    cv2.imwrite("tar_original.png", mask)
    



