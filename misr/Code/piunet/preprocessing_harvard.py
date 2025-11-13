import os
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


def load_dataset_must(base_dir, band):
    folders = []
    for subfolder in os.listdir(base_dir):
        for subsubfolder in os.listdir(os.path.join(base_dir, subfolder)):
            folders.append(os.path.join(base_dir, subfolder, subsubfolder))
    print(folders)

    X = []
    y = []
    for imgset in tqdm(folders):
        LRs = sorted(glob(os.path.join(imgset, band, "lr_3x", "*.png")))
        LR = []

        for i, img in enumerate(LRs):
            LR.append(cv2.imread(img, cv2.IMREAD_UNCHANGED))

        if len(LR) < 2:
            continue

        X.append(np.stack(LR, axis=2))

        y.append(np.expand_dims(cv2.imread(os.path.join(imgset, band, "hr.png"), cv2.IMREAD_GRAYSCALE), axis=2))

    X_flat = np.concatenate([x.flatten() for x in X])
    print(f"X: mean={np.mean(X_flat)}, stdev={np.std(X_flat)}")
    y_flat = np.concatenate([x.flatten() for x in y])
    print(f"y: mean={np.mean(y_flat)}, stdev={np.std(y_flat)}")

    return X, y


def register_dataset(X):
    X_reg = []

    for i in tqdm(range(len(X))):
        img_reg = register_imgset(X[i])
        X_reg.append(img_reg)

    return X_reg


def register_imgset(imgset):
    ref = imgset[..., 0]
    imgset_reg = np.empty(imgset.shape)

    for i in range(imgset.shape[-1]):
        x = imgset[..., i]

        s, _, _ = phase_cross_correlation(ref, x)
        # print(s)
        x = shift(x, s, mode='reflect')
        imgset_reg[..., i] = x

    return imgset


if __name__ == "__main__":
    x, y = load_dataset_must("c:\\datasets\\harvard_s2\\", "b8")
    x = register_dataset(x)

    dataset_output_dir = 'c:\\datasets\\harvard_s2\\'

    np.savez(os.path.join(dataset_output_dir, "mus2_x"), *x)
    np.savez(os.path.join(dataset_output_dir, "mus2_y"), *y)

    print("Done")
