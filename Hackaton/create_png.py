import numpy as np
from tqdm import tqdm
import argparse
import os
import cv2
from utils.XYZ_to_SRGB import XYZ_TO_SRGB


def create_pngs(
        input_path: str,
        output_path: str,
    ) -> None:
    
    os.makedirs(output_path, exist_ok=True)
    files = os.listdir(input_path)
    for filename in tqdm(files):
        if not filename.endswith('gt.npy'):
            continue
        gt = np.load(os.path.join(input_path, filename),  allow_pickle=True)
        gt_xyz  = gt.item().get('xyz')
        SRGB = XYZ_TO_SRGB()
        sRGB_img = SRGB.XYZ_to_sRGB(gt_xyz)
        sRGB_img = sRGB_img * 255
        sRGB_img = np.clip(sRGB_img, 0, 255)
        sRGB_img = sRGB_img.astype(np.uint8)
        sRGB_img = cv2.cvtColor(sRGB_img, cv2.COLOR_RGB2BGR)
        img_name = os.path.basename(filename).split('.')[0]  + '.png'
        img_name = os.path.join(output_path, img_name)
        cv2.imwrite(img_name, sRGB_img)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the quality of the predicted image.')
    parser.add_argument(
        'input_path', 
        type=str,
        help='The path to the predicted images (in png format).'
    )
    parser.add_argument(
        'output_path', 
        type=str,
        help='The path to the predicted images (in png format).'
    )
    return parser.parse_args()


def main():
    """The main function."""
    args = parse_args()
    create_pngs(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
