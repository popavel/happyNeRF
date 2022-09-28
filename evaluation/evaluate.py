import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'evaluate.yaml',
                    'the path of config file')

from tqdm import tqdm
import skimage
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    dataset_path = cfg['dataset']['path']
    actual = cfg['dataset']['actual']
    expected = cfg['dataset']['expected']
    masks = cfg['dataset']['masks']

    actual_path = os.path.join(dataset_path, actual)
    expected_path = os.path.join(dataset_path, expected)
    masks_path = os.path.join(dataset_path, masks)
    output_file_name = actual

    run_evaluation(actual_path, expected_path, masks_path, output_file_name)


def evaluate_lpips(data_path_list):
    print(' LPIPS...')

    lpips_avg = 0
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    for (actual_path, expected_path, mask_path) in tqdm(data_path_list):
        img_actual, img_expected = get_images(actual_path, expected_path, mask_path)

        img_actual_float = img_actual.astype(dtype='float32') / 255.0
        img_expected_float = img_expected.astype(dtype='float32') / 255.0
        img_actual_float = 2 * img_actual_float - 1
        img_expected_float = 2 * img_expected_float - 1

        img_actual_float = np.moveaxis(img_actual_float, -1, 0)
        img_expected_float = np.moveaxis(img_expected_float, -1, 0)
        img_actual_float = img_actual_float[None, ...]
        img_expected_float = img_expected_float[None, ...]

        t_img_actual = torch.from_numpy(img_actual_float)
        t_img_expected = torch.from_numpy(img_expected_float)

        lpips_single_img = lpips(t_img_actual, t_img_expected)
        lpips_avg += lpips_single_img

    print(' LPIPS finished.')

    return lpips_avg / len(data_path_list)


def resize_images(img_actual, img_expected):
    img_actual_resized = img_actual
    img_expected_resized = img_expected

    img_actual_height, img_actual_width, _ = img_actual.shape
    img_expected_height, img_expected_width, _ = img_expected.shape

    if img_actual_height < img_expected_height:
        # should be the case
        y_scale = img_actual_height / img_expected_height
        x_scale = img_actual_width / img_expected_width
        img_expected_resized = cv2.resize(img_expected, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)
    elif img_actual_height > img_expected_height:
        y_scale = img_expected_height / img_actual_height
        x_scale = img_expected_width / img_actual_width
        img_actual_resized = cv2.resize(img_actual, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)
    # else: no resizing

    return img_actual_resized, img_expected_resized


def get_images(actual_path, expected_path, mask_path):
    img_actual = cv2.imread(actual_path)
    img_expected = cv2.imread(expected_path)
    mask = cv2.imread(mask_path)

    mask_bg = mask < 1
    img_expected[mask_bg] = 255

    img_actual_resized, img_expected_resized = resize_images(img_actual, img_expected)

    return img_actual_resized, img_expected_resized


def evaluate_ssim(data_path_list):
    print(' SSIM...')

    ssim_avg = 0

    for (actual_path, expected_path, mask_path) in tqdm(data_path_list):
        img_actual, img_expected = get_images(actual_path, expected_path, mask_path)
        ssim = skimage.metrics.structural_similarity(img_actual, img_expected, channel_axis=2)
        ssim_avg += ssim

    print(' SSIM finished.')

    return ssim_avg / len(data_path_list)


def evaluate_psnr(data_path_list):
    print(' PSNR...')

    psnr_avg = 0

    for (actual_path, expected_path, mask_path) in tqdm(data_path_list):
        img_actual, img_expected = get_images(actual_path, expected_path, mask_path)
        psnr = skimage.metrics.peak_signal_noise_ratio(img_expected, img_actual)
        psnr_avg += psnr

    print(' PSNR finished.')

    return psnr_avg / len(data_path_list)


def run_evaluation(actual_path, expected_path, masks_path, output_file_name):
    print('Starting valuation...')

    file_extension_0 = 'jpg'
    file_extension_1 = 'png'
    actual_path_list = glob.glob(os.path.join(actual_path, '*.' + file_extension_1))
    expected_path_list = glob.glob(os.path.join(expected_path, '*.' + file_extension_0))
    masks_path_list = glob.glob(os.path.join(masks_path, '*.' + file_extension_0))

    assert len(actual_path_list) == len(expected_path_list) and len(expected_path_list) == len(masks_path_list), \
        f'different number of ' \
        f'actual ({len(actual_path_list)}), ' \
        f'expected ({len(expected_path_list)})  and ' \
        f'mask ({len(masks_path_list)}) ' \
        f'images to compare'

    data_path_list = list(zip(actual_path_list, expected_path_list, masks_path_list))

    lpips_avg = evaluate_lpips(data_path_list)
    ssim_avg = evaluate_ssim(data_path_list)
    psnr_avg = evaluate_psnr(data_path_list)

    with open(output_file_name + '.txt', 'w') as f:
        f.write(output_file_name + '\n' +
                f'    lpips_avg: ({lpips_avg})\n' +
                f'    ssim_avg: ({ssim_avg})\n' +
                f'    psnr_avg: ({psnr_avg})\n')

    print('Evaluation finished.')
    print('Results:')
    print(f'    lpips_avg: ({lpips_avg})')
    print(f'    ssim_avg: ({ssim_avg})')
    print(f'    psnr_avg: ({psnr_avg})')


if __name__ == '__main__':
    app.run(main)
