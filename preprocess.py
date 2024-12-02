import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def equalize_histo(img):
    r_image, g_image, b_image = cv2.split(img)

    r_image_eq = cv2.equalizeHist(r_image)
    g_image_eq = cv2.equalizeHist(g_image)
    b_image_eq = cv2.equalizeHist(b_image)

    image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
    return image_eq

def get_image(img_path):
        image = equalize_histo(cv2.imread(img_path)[...,::-1])/255.
        return image


def get_transform(mode='train'):
        if mode =='train':
                return A.Compose([
                        A.Resize(width=224, height=224),
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.ShiftScaleRotate(shift_limit=(-0.05,0.05),scale_limit=(-0.1,0.2),rotate_limit=(-10,10),p=0.5),
                        A.Normalize(mean=0.5,std=0.5,max_pixel_value=1.0, always_apply=True, p=1.0),
                        ToTensorV2(transpose_mask=True)])
        else:
                return A.Compose([
                        A.Resize(width=224, height=224),
                        A.Normalize(mean=0.5,std=0.5,max_pixel_value=1.0, always_apply=True, p=1.0),
                        ToTensorV2(transpose_mask=True)])
        