from typing import Sequence

import cv2
import numpy as np
from scipy.signal import medfilt2d, convolve2d
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
from skimage.transform import resize
import scipy.ndimage as sni

import os
# os.environ['PATH'] += ";c:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.35.32215\\bin\\Hostx64\\x64\\"
os.environ['PATH'] += ";c:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64\\"

def registration(images, scale) -> Sequence[np.ndarray]:
    T = [np.zeros((2,))]
    fixed = images[:, :, 0]

    for i in range(1, images.shape[2]):
        shift, error, diffphase = phase_cross_correlation(fixed, images[:, :, i], upsample_factor=int(scale * 4))
        T.append(shift)

    return T


def gaussian_kernel(size, sigma):
    blur = np.zeros((size, size))
    blur[blur.shape[0] // 2, blur.shape[0] // 2] = 1
    return cv2.GaussianBlur(blur, (size, size), sigma)


def decimation(img, factor):
    return img[::factor, ::factor]


def shift(img, m, l):
    return np.roll(np.roll(img, m, axis=1), l, axis=0)

import cupy as cp


def MedianAndShift(LR: np.ndarray, D: np.ndarray, HRsize: tuple, Dres: int) -> tuple:
    # LR: WxHxN
    # D: Nx2

    # Allocate high resolution image
    LR_ = cp.asarray(LR)
    Z = cp.empty((HRsize[0], HRsize[1], LR_.shape[2]))
    Z[:] = cp.nan

    shifts = np.abs(D.astype(int) + (Dres // 2))
    for i in range(LR_.shape[2]):
        lr_i = LR_[:, :, i]
        lr_i_shift = shifts[i, :]
        lr_i = cp.roll(lr_i, lr_i_shift)
        Z[lr_i_shift[0]::Dres, lr_i_shift[1]::Dres, i] = lr_i

    A = cp.count_nonzero(~np.isnan(Z), axis=2)
    Z = cp.nanmedian(Z, 2)
    A = cp.sqrt(A)

    return Z, A


def srr(images: np.ndarray, alpha: float, beta: float, _lambda: float, sigma: float, P: int, niter: int,
        scale: int) -> np.ndarray:
    shift_matrices = registration(images, scale)
    shifts = (np.stack([sm for sm in shift_matrices]) * scale).round()
    inverse_shifts = -shifts

    # def MedianAndShift(LR: np.ndarray, D: np.ndarray, HRsize: tuple, Dres: int) -> tuple:
    HRsize = (scale * images.shape[0], scale * images.shape[1])
    Z, A = MedianAndShift(images, shifts, HRsize, scale)

    psf = gaussian_kernel(5, sigma)
    psf_transposed = psf.transpose()

    X0: np.ndarray = resize(np.median(images, axis=2), HRsize, order=1)
    X: np.ndarray = X0
    z = np.copy(X)

    dest_size = (scale * images.shape[0], scale * images.shape[1])

    K = images.shape[2]
    for i in range(niter):
        print(f"{i}")

        gradients_sum = np.zeros_like(X)
        for k in range(K):
            im_shifted = X#warp(X, inverse_shift_matrices[k], output_shape=dest_size)
            gradient_block = convolve2d(im_shifted, psf, mode='same')
            gradient_block = decimation(gradient_block, scale)
            gradient_block = gradient_block - images[:, :, k]
            gradient_block = np.sign(gradient_block)
            gradient_block = resize(gradient_block, dest_size, order=1)
            gradient_block = convolve2d(gradient_block, psf_transposed, mode='same')
            # gradient_block = warp(gradient_block, shift_matrices[k], output_shape=dest_size)

            gradients_sum = gradients_sum + gradient_block

        # Regularization term
        regularization = np.zeros_like(X)
        for l in range(P + 1):
            for m in range(P + 1):
                a_pow_absl_absm = (alpha ** (np.abs(m) + np.abs(l)))
                reg_block = X - shift(X, m, l)
                reg_block = np.sign(reg_block)
                reg_block = reg_block - shift(reg_block, -m, -l)
                reg_block = a_pow_absl_absm * reg_block
                regularization = regularization + reg_block

        gradient = gradients_sum + _lambda * regularization

        # Update
        X = X - beta * gradient

        # Check for convergence
        norm2 = np.linalg.norm(gradient)
        if norm2 < 10:
            break

    return X


def _loadz(path: str):
    res = np.load(path)
    return [res[f].astype(np.float32) for f in res.files]


x_LR = _loadz("C:\\projekty\\piunet\\Dataset\\mus2_x.npz")
x_HR = _loadz("C:\\projekty\\piunet\\Dataset\\mus2_y.npz")

niter = 2
scale = 3
beta = 1
_lambda = 0.05
alpha = 0.6
P = 3

for i in range(10):
    result = srr(x_LR[i], alpha, beta, _lambda, 1, P, niter, scale)

dir = "c:\\experiments\\_do_wyrzucenia\\"
cv2.imwrite(f"{dir}lr0.png", x_LR[0][:, :, 0].astype(np.uint16))
cv2.imwrite(f"{dir}hr.png", x_HR[0][:, :, 0].astype(np.uint8))
cv2.imwrite(f"{dir}result.png", result.astype(np.uint16))

print('x')
