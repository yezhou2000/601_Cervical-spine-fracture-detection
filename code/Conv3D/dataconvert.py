import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
from PIL import Image
# from dipy.denoise.nlmeans import nlmeans
# from dipy.denoise.noise_estimate import estimate_sigma
from pydicom.pixel_data_handlers import apply_voi_lut
from kaggle_volclassif.utils import interpolate_volume
from skimage import exposure


PATH_DATASET = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
def convert_volume(dir_path: str, out_dir: str = "train_volumes", size=(224, 224, 224)):
    ls_imgs = glob.glob(os.path.join(dir_path, "*.dcm"))
    ls_imgs = sorted(ls_imgs, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    imgs = []
    for p_img in ls_imgs:
        dicom = pydicom.dcmread(p_img)
        img = apply_voi_lut(dicom.pixel_array, dicom)
        img = cv2.resize(img, size[:2], interpolation=cv2.INTER_LINEAR)
        imgs.append(img.tolist())
    vol = torch.tensor(imgs, dtype=torch.float32)

    vol = (vol - vol.min()) / float(vol.max() - vol.min())
    vol = interpolate_volume(vol, size).numpy()

    # https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html
    vol = exposure.equalize_adapthist(vol, kernel_size=np.array([64, 64, 64]), clip_limit=0.01)
    # vol = exposure.equalize_hist(vol)
    vol = np.clip(vol * 255, 0, 255).astype(np.uint8)

    path_npz = os.path.join(out_dir, f"{os.path.basename(dir_path)}.npz")
    np.savez_compressed(path_npz, vol)