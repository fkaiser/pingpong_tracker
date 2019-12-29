import cv2
import numpy as np
from matplotlib import pyplot as plt


class motionTracker:

    def __init__(self, image_path, n_particles=10):
        self.n_particles = n_particles
        self.image_path = image_path
        self.image = self.load_image(self.image_path)
        self.target_ROI = self.get_ROI(self.image)
        self.get_target_hist()

    def get_ROI(self, img):
        # Select ROI
        fromCenter = False
        r = cv2.selectROI(img, fromCenter)

        # Crop image
        height = int(r[1] + r[3]) - int(r[1])
        width = int(r[0] + r[2]) - int(r[0])
        im_crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        target_ROI = {'image': im_crop, 'image_gray':
                      self.convert_to_grayscale(im_crop),
                      'height': height, width: 'width'}
        return target_ROI

    def load_image(self, image_path):
        im = cv2.imread(image_path)
        return im

    def get_target_hist(self):
        im_gray = self.convert_to_grayscale(self.image)
        plt.subplot(121), plt.imshow(self.target_ROI['image_gray'], 'gray')
        plt.subplot(122)
        self.target_hist = dict()
        self.target_hist['values'], self.target_hist['bins'], _ = plt.hist(
            self.target_ROI['image_gray'].ravel(), 50, [0, 256])
        plt.xlim(0, 256)
        plt.show()

    @staticmethod
    def convert_to_grayscale(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def init_particles(self):
        pass



def main():
    image_path = 'image0203.png'
    ping_pong_tracker = motionTracker(image_path=image_path)


if __name__ == '__main__':
    main()
