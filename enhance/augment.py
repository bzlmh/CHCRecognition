import os
import cv2
import sys
import random
import numpy as np
from movement import distort, stretch, perspective

def cv_imread(filepath):
    """
    Equivalent to cv2.imread
    :param filepath: Path of the image to be read, e.g.,
    """
    cv_imr = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_imr

def cv_imwrite(filepath, img):
    """
    Equivalent to cv2.imwrite
    :param filepath: Path to save the image, e.g.,
    :param img: Image to be saved, a three-dimensional array
    """
    cv_imw = cv2.imencode('.jpg', img)[1].tofile(filepath)
    return cv_imw

if __name__ == '__main__':
    directory = r"data2/train_data"
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            for file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file)
                print(image_path)
                im = cv_imread(image_path)
                im1 = cv2.resize(im, (200, 64))
                im2 = cv2.resize(im, (1000, 100))

                for i in range(3):
                    augmentation_type = random.choice(['distort', 'stretch', 'perspective'])

                    if augmentation_type == 'distort':
                        distort_img = distort(im1, 4)
                        distort_img = cv2.resize(distort_img, (64, 64))
                        path = os.path.join(folder_path, f'distort_img_{i}.jpg')
                        cv_imwrite(path, distort_img)

                    elif augmentation_type == 'stretch':
                        stretch_img = stretch(im2, 4)
                        stretch_img = cv2.resize(stretch_img, (64, 64))
                        path = os.path.join(folder_path, f'stretch_img_{i}.jpg')
                        cv_imwrite(path, stretch_img)

                    elif augmentation_type == 'perspective':
                        perspective_img = perspective(im1)
                        perspective_img = cv2.resize(perspective_img, (64, 64))
                        path = os.path.join(folder_path, f'perspective_img_{i}.jpg')
                        cv_imwrite(path, perspective_img)
