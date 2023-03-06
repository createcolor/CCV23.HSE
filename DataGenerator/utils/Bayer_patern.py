import numpy as np
import cv2


class Bayer:
    def __init__(self):
        #   G R
        #   B G
        self.Bayer_PATTERN = "GRBG"

    def sRGB_to_Bayer(self, sRGB):
        H, W, _ = sRGB.shape
        (B, G, R) = cv2.split((sRGB * 255).astype(np.uint16))

        bayer = np.zeros((H, W), np.uint8)

        bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
        bayer[0::2, 1::2] = R[0::2, 1::2]  # top right
        bayer[1::2, 0::2] = B[1::2, 0::2]  # bottom left
        bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right

        return bayer

    def add_gaussian_noise(self, bayer, mean=0, sigma=20):
        gaussian_noise = np.random.normal(mean, sigma, bayer.shape)
        gaussian_noise = gaussian_noise.reshape(bayer.shape)
        noisy_image = bayer + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint16)
        return noisy_image
