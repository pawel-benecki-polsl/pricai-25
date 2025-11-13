import os
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


def load_dataset_must(base_dir, band):
    imgsets = sorted(glob(os.path.join(base_dir, "*")))

    X = []
    y = []
    for imgset in tqdm(imgsets):
        LRs = sorted(glob(os.path.join(imgset, band, "lrs", "*.jp2")))

        LR = []

        for i, img in enumerate(LRs):
            LR.append(cv2.imread(img, cv2.IMREAD_UNCHANGED))

        X.append(np.stack(LR, axis=2))

        y.append(np.expand_dims(cv2.imread(os.path.join(imgset, "hr_resized", "mul_band_6.tiff"), cv2.IMREAD_GRAYSCALE), axis=2))

    X_flat = np.concatenate([x.flatten() for x in X])
    print(f"X: mean={np.mean(X_flat)}, stdev={np.std(X_flat)}")
    y_flat = np.concatenate([x.flatten() for x in y])
    print(f"y: mean={np.mean(y_flat)}, stdev={np.std(y_flat)}")

    import matplotlib.pyplot as plt

    plt.hist(y_flat, bins=256)
    plt.title("Histogram y_flat")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(X_flat, bins=256)
    plt.title("Histogram X_flat")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

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
    x, y = load_dataset_must("c:\\datasets\\mus2\\image_data\\", "b8")
    # x = register_dataset(x)
    #
    # dataset_output_dir = '../../Dataset/'
    #
    # np.savez(os.path.join(dataset_output_dir, "mus2_x"), *x)
    # np.savez(os.path.join(dataset_output_dir, "mus2_y"), *y)
    #
    # print("Done")
