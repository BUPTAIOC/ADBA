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


def save_and_resize_image(image_tensor, file_path):# save adverserial examples
    utils.save_image(image_tensor, file_path)
    img = Image.open(file_path)
    img = img.resize((320, 320), Image.NEAREST)
    img.save(file_path, format='BMP')  


def concatenate_images(image_paths):# save adverserial examples
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
def save_images(original_image, adversarial_image, adversarial_v, plustring):# save adverserial examples
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
# a,b,c,d
a_hat = 0.03133292769944518
b_hat = 3.0659694403842903
c_hat = 0.16755646970211466
d_hat = 0.13403850261898806



# 
def func_rho(r, a, b, c, d):
    return a / ((r + d) ** b) + c


# 
def find_midK_of_k1k2(k1, k2): # find medianpoint of func_rho
    Sk1k2, error = quad(func_rho, k1, k2, args=(a_hat, b_hat, c_hat, d_hat))
    low, high, mid = k1, k2, 0
    Sk1mid = None
    while high - low > 1.0 / 600:
        mid = (low + high) / 2
        Sk1mid, error = quad(func_rho, k1, mid, args=(a_hat, b_hat, c_hat, d_hat))
        if Sk1mid < Sk1k2 / 2:
            low = mid
        else:
            high = mid
    return mid


def next_ADB(r1, r2, aim_r, max_r, mod): #decide next ADB using func_rho()
    if mod == 0:#ADBA
        return (r1 + r2) / 2.0
    if mod == 1:#ADBA-md
        k1, k2 = r1 / max_r, r2 / max_r
        kmid = find_midK_of_k1k2(1-k2, 1-k1)
        median = (1-kmid) * max_r
        return median