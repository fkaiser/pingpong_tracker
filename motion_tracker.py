import cv2
import numpy as np
from matplotlib import pyplot as plt


class motionTracker:

    def __init__(self, image_path, n_particles=10, sigma_init_pos=20,
                 sigma_init_vel=1, process_noise_sigma=10, n_steps=100, n_states=4):
        self.n_particles = n_particles
        self.sigma_init_pos = sigma_init_pos
        self.sigma_init_vel = sigma_init_vel
        self.process_noise_sigma = process_noise_sigma
        self.n_steps = n_steps
        self.n_states = n_states
        self.compute_init_variance()
        self.image_path = image_path
        self.image = self.load_image(self.image_path)
        self.target_ROI = self.get_ROI(self.image)
        self.get_target_hist()
        self.init_particles()
        self.show_particles()

    def compute_init_variance(self):
        self.P0_init = np.diag([self.sigma_init_pos, self.sigma_init_pos,
                                self.sigma_init_vel, self.sigma_init_vel])

    def get_ROI(self, img):
        # Select ROI
        fromCenter = False
        r = cv2.selectROI(img, fromCenter)

        # Crop image
        width = int(r[1] + r[3]) - int(r[1])
        height = int(r[0] + r[2]) - int(r[0])
        im_crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        target_ROI = {'image': im_crop, 'image_gray':
                      self.convert_to_grayscale(im_crop),
                      'height': height, width: 'width',
                      'target_midpoint_x': int(r[0]) + height / 2,
                      'target_midpoint_y': int(r[1]) + width / 2}
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
        self.X_p = np.zeros((self.n_steps, self.n_states, self.n_particles))
        mean_init = np.array([[self.target_ROI['target_midpoint_x']],
                              [self.target_ROI['target_midpoint_y']],
                              [0],
                              [0]])
        self.X_p[0, :, :] = mean_init + \
            np.dot(self.P0_init, np.random.normal(
                size=(self.n_states, self.n_particles)))

    def show_particles(self):
        # Radius of circle
        radius = 10

        # Red color in BGR
        color = (0, 0, 255)
        center_list = [(int(self.X_p[0, 0, i]), int(self.X_p[0, 1, i]))
                       for i in range(self.n_particles)]
        for center_point in center_list:
            cv2.circle(self.image, center_point,
                       radius=radius, color=color)

        window_name = 'particles'

        # Displaying the image
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)


def main():
    image_path = 'image0203.png'
    ping_pong_tracker = motionTracker(image_path=image_path)


if __name__ == '__main__':
    main()
