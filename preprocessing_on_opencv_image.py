import cv2
import numpy as np
import math

IMAGENET_CHANNEL_MEANS = [0.485, 0.456, 0.406]
IMAGENET_CHANNEL_STDS = [0.229, 0.224, 0.225]


def center_crop(img, crop_w, crop_h):
    img_h, img_w = img.shape[:2]
    top = (img_h - crop_h) // 2
    left = (img_w - crop_w) // 2
    return img[top:top + crop_h, left: left + crop_w, :]


def resize_keep_aspect(img, keep_min=True, min_resize=None):
    if keep_min and min_resize is not None:
        img_h, img_w, _ = img.shape
        smaller_size = min(img_h, img_w)
        ratio = min_resize / smaller_size
        h = int(img_h * ratio)
        w = int(img_w * ratio)
        return cv2.resize(img, (w, h))
    else:
        raise NotImplementedError


def normalize(img, mean, std):
    img = img / 255.0
    img -= np.array(mean)
    img /= np.array(std)
    return img


def preprocess(img_path):
    img = cv2.imread(img_path)
    crop_ratio = 0.875
    min_resize = int(math.ceil(224 / crop_ratio))

    img = resize_keep_aspect(img, keep_min=True, min_resize=min_resize)
    img = center_crop(img, 224, 224)
    img = normalize(img, IMAGENET_CHANNEL_MEANS, IMAGENET_CHANNEL_STDS)
    return img[None, :, :, :]
