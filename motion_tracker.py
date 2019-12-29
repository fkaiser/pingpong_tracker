import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_ROI(img):
    # Select ROI
    fromCenter = False
    r = cv2.selectROI(img, fromCenter)

    # Crop image
    height = int(r[1]+ r[3]) - int(r[1])
    width = int(r[0] + r[2]) - int(r[0])
    im_crop = img[int(r[1]):int(r[1]+ r[3]), int(r[0]):int(r[0] + r[2])]
    target_ROI = {'image': im_crop, 'height': height, width: 'width'}
    return target_ROI


def convert_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def get_target_hist(image_path):
    im = cv2.imread('image0203.png')
    im_gray = convert_to_grayscale(im)
    target_ROI = get_ROI(im_gray)
    plt.subplot(121), plt.imshow(target_ROI['image'], 'gray')
    plt.subplot(122)
    target_hist = dict()
    target_hist['values'], target_hist['bins'], _ = plt.hist(target_ROI['image'].ravel(),50,[0,256])
    plt.xlim(0,256)
    plt.show()
    return hist_gray


def main():
    image_path = 'frames/cellphone/ground/image0203.png'
    target_hist = get_target_hist(image_path)


if __name__ == '__main__':
    main()
