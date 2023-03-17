# Camera pipeline hackaton

## Intro

This folder is devoted to your homework and it is a small challenge for you.
Eeach should develop its own camera pipeline. For this purpose we provide you: 
1. dataset 
2. augmentation script 
3. quality evaluation metric

Your goal is to develop the best pipeilne, which input is RAW image captured by Canon 600d and output is ideal XYZ. 

Score is sum of PSNR and SSIM.

To test your sollution we will generate new hidden images with a unique random seed and compare your XYZ prediction with GT. First five solution will have 10 points automatically, while others will defend their pipelines.

## Dataset

A seria of [hyperspectral images](https://truenascloud.duckdns.org:9998/s/NxiNpoeYC4xpMTD) have been prepared. Based on these images we generate Canon 600D camera RAW images and XYZ images. 

## Code

1. Use Python 3.10
2. Install venv
3. Create venv
```bash
$ python -m venv venv
```
4. Activate venv 
5. Install requirements:
```bash
$ pip install -r requirements.txt
```

### Generate Data

Process to generate dataset.

```bash
$ python dataset_generator.py -h
Process to generate dataset.

positional arguments:
  input_path      Path to the directory containing spectral images. (in str format)
  output_path     Path to save ground-truth and sample. (in str format)
  mean            Mean it is mean of noise.  (in float format)  advance
  sigma           Sigma it is sigma of noise. (in float format) advance 
  seed            Random seed for light and noise        

optional arguments:
  -h, --help         show this help message and exit
```
Example:

```bash
$ python dataset_generator.py --input_path ./data_example/input_data/ --output_path ./data_example/output_generated_data/ --seed 40
Processing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Processing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Data had created successfully for spectral image: ./data_example/input_data/2019-08-25_006.h5
```

All data will be saved in `output_path` directory in our example, it will be saved in `./data_example/output_generated_data/`.
The generated data will be in `.npy`format and prefix `gt` mean Ground truth and `sample` is generated data using 
`Canon 600D camera`

### Evaluator

This is a evaluator for Image Sigral Processing pieplines.

```bash
$ python evaluator.py -h
Evaluate the quality of Image Sigral Processing piepline.

positional arguments:
  predcted_path      The path to the predicted images (in png format).
  ground_truth_path  The path to the ground truth images (in png format).

optional arguments:
  -h, --help         show this help message and exit
```

Example:

```bash
$ python evaluator.py /data/predicted /data/ground_truth
100%|███████| 56/56 [00:00<00:00, 151.12it/s]
PSNR: 32
SSIM: 0.27
Score: 0.29
```
