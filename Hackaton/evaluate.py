import numpy as np
from tqdm import tqdm
from typing import Tuple
import argparse
import os
import cv2


def psnr(
        img1: np.ndarray, 
        img2: np.ndarray, 
        max_value: float = 255
    ) -> float:
    """Calculate the PSNR between two images.

    The PSNR is the logarithmic mean of the squared error between the two images.
    The error is calculated per channel and then averaged over all channels.

    Parameters:
        img1 (numpy.ndarray): The first image, as a 2D array.
        img2 (numpy.ndarray): The second image, as a 2D array of the same shape as img1.
        max_value (float): The maximum pixel value of the images (default 255).

    Returns:
        float: PSNR value.
    """
    # Ensure the images are of the same shape
    assert img1.shape == img2.shape, "Images must be of the same shape."

    # Calculate the PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_value) - 10 * np.log10(mse)


def ssim(
        img1: np.ndarray, 
        img2: np.ndarray, 
        max_value: int = 255,
        k1: int = 0.01, 
        k2: int = 0.03,
    ) -> float:
    """Calculates the Structural Similarity Index (SSIM) between two images.
    
    Parameters:
        img1 (numpy.ndarray): The first image, as a 2D array.
        img2 (numpy.ndarray): The second image, as a 2D array of the same shape as img1.
        k1 (float): The constant used to stabilize the SSIM calculation (default 0.01).
        k2 (float): The constant used to stabilize the SSIM calculation (default 0.03).
        max_value (float): The maximum pixel value of the images (default 255).
    
    Returns:
        float: The SSIM between the two images, between -1 and 1.
    """
    # Ensure the images are of the same shape
    assert img1.shape == img2.shape, "Images must be of the same shape."
    
    # Convert the images to floats and scale them to the range [0, 1]
    img1 = img1.astype(np.float64) / max_value
    img2 = img2.astype(np.float64) / max_value
    
    # Compute the mean, variance, and covariance of the images
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = np.cov(img1.flat, img2.flat)[0, 1]
    
    # Calculate the SSIM
    C1 = (k1 * max_value) ** 2
    C2 = (k2 * max_value) ** 2
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / denominator
    
    return ssim


def evaluate(
        predcted_path: str,
        ground_truth_path: str,
    ) -> Tuple[float, float, float]:
    """Evaluate the quality of Image Sigral Processing piepline.

    Parameters:
        predcted_path (str): The path to the predicted images (in png format).
        ground_truth_path (str): The path to the predicted images (in png format).
    
    Returns:
        Tuple[float, float, float]: The PSNR, SSIM, and Score between the images.
    """
    psnr_list = []
    ssim_list = []
    predcted_images = [x for x in os.listdir(predcted_path) if x.lower().endswith('png')]
    ground_truth_images = [x for x in os.listdir(predcted_path) if x.lower().endswith('png')]
    assert len(predcted_images) == len(ground_truth_images), "The number of predicted images must match the number of ground truth images."
    
    for img_name in tqdm(predcted_images):
        assert img_name in ground_truth_images, f"Image {img_name} is not in the ground truth directory."

        pred = cv2.imread(
            os.path.join(predcted_path, img_name), 
            cv2.IMREAD_GRAYSCALE
        )
        gt = cv2.imread(
            os.path.join(ground_truth_path, img_name),
            cv2.IMREAD_GRAYSCALE
        )
        psnr_list.append(psnr(pred, gt))
        ssim_list.append(ssim(pred, gt))

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(psnr_list) * np.mean(ssim_list)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the quality of the predicted image.')
    parser.add_argument(
        'predcted_path', 
        type=str,
        help='The path to the predicted images (in png format).'
    )
    parser.add_argument(
        'ground_truth_path', 
        type=str,
        help='The path to the predicted images (in png format).'
    )
    return parser.parse_args()


def main():
    """The main function."""
    args = parse_args()
    psnr, ssim, score = evaluate(args.predcted_path, args.ground_truth_path)
    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {ssim:.2f}")
    print(f"Score: {score:.2f}")


if __name__ == '__main__':
    main()
