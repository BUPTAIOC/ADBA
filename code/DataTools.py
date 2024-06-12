# coding:utf-8
import os
import subprocess
import random
from model import MNIST, CIFAR10
from PIL import Image
from torchvision import utils, transforms
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import fsolve
import torchvision.datasets as dsets
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


def save_and_resize_image(image_tensor, file_path):
    utils.save_image(image_tensor, file_path)
    img = Image.open(file_path)
    img = img.resize((320, 320), Image.NEAREST)
    img.save(file_path, format='BMP')  # 保存为位图格式


def concatenate_images(image_paths):
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img

""" """
def save_images(original_image, adversarial_image, adversarial_v, plustring):
    folder_name = "adversarial_samples"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    original_file = os.path.join(folder_name, f"ORIG_{plustring}.bmp")
    adversarial_v_file = os.path.join(folder_name, f"ADVV_{plustring}.bmp")
    adversarial_file = os.path.join(folder_name, f"ADVO_{plustring}.bmp")
    combined_file = os.path.join(folder_name, f"COMB_{plustring}.bmp")

    # Save and resize images in BMP format
    save_and_resize_image(original_image, original_file)
    save_and_resize_image(adversarial_image, adversarial_file)

    # Save perturbation image in BMP format
    # adversarial_v = adversarial_image - original_image
    save_and_resize_image(adversarial_v, adversarial_v_file)

    # Concatenate and save combined image in BMP format
    combined_image = concatenate_images([adversarial_v_file, original_file, adversarial_file])
    combined_image.save(combined_file, format='BMP')
    os.remove(adversarial_v_file)
    os.remove(original_file)
    os.remove(adversarial_file)
    return combined_file


def open_image(image_path):
    if os.name == 'nt':  # Windows
        os.startfile(image_path)
    elif os.name == 'posix':  # macOS and Linux
        subprocess.run(['open', image_path], check=True)

##################################################################################
# 给定参数 cifar10/Imagenet

a_hat = 0.03133292769944518
b_hat = 3.0659694403842903
c_hat = 0.16755646970211466
d_hat = 0.13403850261898806
"""
# 给定参数 mnist-0.3
a_hat = 0.39452608479382933
b_hat = 0.7585850099734531
c_hat = 0.06301194520345259
d_hat = 0.04770313750921767
"""
"""
# 给定参数 mnist-0.2
a_hat = 0.018364609122357524
b_hat = 2.295602162862136
c_hat = 0.021719135417420987
d_hat = 0.03725511515174054
"""


# 定义函数 y
def func_y(r, a, b, c, d):
    return a / ((r + d) ** b) + c


# 为了使用 fsolve，需要定义一个差函数
def find_midK_of_k1k2(k1, k2):
    Sk1k2, error = quad(func_y, k1, k2, args=(a_hat, b_hat, c_hat, d_hat))
    low, high, mid = k1, k2, 0
    Sk1mid = None
    while high - low > 1.0 / 600:
        mid = (low + high) / 2
        Sk1mid, error = quad(func_y, k1, mid, args=(a_hat, b_hat, c_hat, d_hat))
        if Sk1mid < Sk1k2 / 2:
            low = mid
        else:
            high = mid
    return mid


def next_binary_rref(r1, r2, aim_r, max_r, mod):
    if mod == 0:
        return (r1 + r2) / 2.0
    if mod == 1:
        k1, k2 = r1 / max_r, r2 / max_r
        kmid = find_midK_of_k1k2(1-k2, 1-k1)
        median = (1-kmid) * max_r
        return median


def main():
    # 假设的输入值
    l = 0
    r = 1
    """
    P = 0.93  # 在[a, b]范围内的置信度
    # 计算标准差
    calculated_sigma = find_sigma_for_symmetric_interval(r, r, P)
    a = find_a_for_symmetric_interval(r, calculated_sigma, 0.5 * P)
    x = r - a
    print("Calculated Sigma:", calculated_sigma)
    print("Calculated Integral P/2:", normal_distribution_integral(r, calculated_sigma, l, r))
    print("Calculated a:", a)
    print("Calculated x:", x)
    print("real Integral P:", P)
    print("Calculated Integral P/4:", normal_distribution_integral(r, calculated_sigma, l, x))"""
    mid = find_midK_of_k1k2(0, 1)
    print("real Integral mid:", mid)

    low, high = l, r
    for i in range(9):
        # mid = find_mid_of_lowhigh(E=r, low=low, high=high, sigma=calculated_sigma)
        mid = find_midK_of_k1k2(low, high)
        """print(f" i_{i}"
              f",\tlow={round(low, 3)}"
              f",\thigh={round(high, 3)}"
              f",\tmid={round(mid, 3)}"
              f",\tk2={round((mid - low) / (high - low), 3)}"
              )"""
        high = mid


if __name__ == "__main__":
    main()
