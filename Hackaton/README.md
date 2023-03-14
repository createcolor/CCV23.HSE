# Data

download data from this link: https://truenascloud.duckdns.org:9998/s/NxiNpoeYC4xpMTD

# Demo


# Evaluator

This is a evaluator for Image Sigral Processing pieplines.

## Installation

```bash
$ pip install -r requirements.txt
```

## Usage

```bash
$ python evaluator.py -h
Evaluate the quality of Image Sigral Processing piepline.

positional arguments:
  predcted_path      The path to the predicted images (in png format).
  ground_truth_path  The path to the ground truth images (in png format).

optional arguments:
  -h, --help         show this help message and exit
```

## Example

```bash
$ python evaluator.py /data/predicted /data/ground_truth
100%|███████| 56/56 [00:00<00:00, 151.12it/s]
PSNR: 32
SSIM: 0.27
Score: 0.29
```