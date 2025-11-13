import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from Code.piunet.metrics.models import load_lpips, Models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cPSNR(sr, hr, hr_map):
    n_clear = np.sum(hr_map)
    diff = hr - sr
    bias = np.sum(diff * hr_map) / n_clear
    cmse = np.sum(np.square((diff - bias) * hr_map)) / n_clear
    cpsnr = -10 * np.log10(cmse)
    return cpsnr


def cSSIM(sr, hr, hr_map):
    n_clear = np.sum(hr_map)
    diff = hr - sr
    bias = np.sum(diff * hr_map) / n_clear
    cssim = ssim((sr + bias) * hr_map, hr * hr_map, data_range=255)
    return cssim


def get_masked_sentinel_image(sentinel_img, wv_img, mask):
    new_shape = (sentinel_img.shape[1], sentinel_img.shape[0])
    mask = cv2.resize(mask.astype(np.uint8), new_shape, interpolation=cv2.INTER_AREA)
    inverse_mask = np.invert(mask.astype(np.bool_)).astype(np.uint8)
    wv_masked = cv2.bitwise_and(wv_img, wv_img, mask=mask)
    resized_masked_sentinel = cv2.bitwise_and(sentinel_img, sentinel_img, mask=inverse_mask)
    resized_sentinel = wv_masked + resized_masked_sentinel
    return resized_sentinel, mask


class MetricsSingleBand:
    def __init__(self):
        self.ssim = ssim
        self.lpips = load_lpips()
        self.lpips_mix = load_lpips(Models.LPIPS_1VOTE_KFS_2VOTES_HUMAN)
        self.lpips_kfs = load_lpips(Models.LPIPS_KFS_20221206)
        self.shift = None
        self.metrics = {
            "PSNR": self.get_psnr,
            "SSIM": self.get_ssim,
            "LPIPS": self.get_lpips,
            "LPIPS_KFS": self.get_lpips_kfs,
            "LPIPS_MIX": self.get_lpips_mix,
            "cPSNR": self.get_cpsnr,
            "cSSIM": self.get_cssim,
            # "cLPIPS": self.get_clpips,
        }

    @staticmethod
    def get_psnr(sr, hr, mask=None):
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        if np.max(sr) > 1:
            sr = sr / 255
        if np.max(hr) > 1:
            hr = hr / 255
        n_clear = np.sum(mask)
        mse = np.sum(((sr - hr) * mask) ** 2) / n_clear
        if mse == 0:
            return 361
        psnr = -10 * np.log10(mse)
        return psnr

    def get_ssim(self, sr, hr, mask):
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        return self.ssim(sr * mask, hr * mask, data_range=255)

    def _get_lpips(self, model, sr, hr, mask):
        def prepare_input_tensor(img):
            img_interp = (img - 127.5) / 127.5
            img_interp = np.expand_dims(img_interp, axis=0)
            img_tensor = torch.tensor(img_interp, device=DEVICE).float()
            return img_tensor

        if mask is not None:
            sr, _ = get_masked_sentinel_image(sr, hr, mask)
        img_1_tensor = prepare_input_tensor(sr)
        img_2_tensor = prepare_input_tensor(hr)
        return float(model(img_1_tensor, img_2_tensor))

    def get_lpips(self, sr, hr, mask):
        return self._get_lpips(self.lpips, sr, hr, mask)

    def get_lpips_kfs(self, sr, hr, mask):
        return self._get_lpips(self.lpips_kfs, sr, hr, mask)

    def get_lpips_mix(self, sr, hr, mask):
        return self._get_lpips(self.lpips_mix, sr, hr, mask)


    def get_cpsnr(self, sr, hr, mask=None, max_shift=3):
        if np.max(sr) > 1:
            sr = sr / 255
        if np.max(hr) > 1:
            hr = hr / 255
        sr_cropped = sr[max_shift:-max_shift, max_shift:-max_shift]
        shape = hr.shape
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        cpsnr_values = []
        shifts = []
        for ii in range(2 * max_shift):
            for jj in range(2 * max_shift):
                ii_end = shape[0] - (2 * max_shift - ii)
                jj_end = shape[1] - (2 * max_shift - jj)
                registered_hr = hr[ii:ii_end, jj:jj_end]
                registered_mask = mask[ii:ii_end, jj:jj_end]
                cpsnr = cPSNR(sr_cropped, registered_hr, registered_mask)
                cpsnr_values.append(cpsnr)
                shifts.append((ii, jj))
        self.shift = shifts[np.argmax(cpsnr_values)]
        return np.max(cpsnr_values)

    def get_cssim(self, sr, hr, mask=None, max_shift=3):
        hr = hr.astype(float)
        sr = sr.astype(float)
        sr_cropped = sr[max_shift:-max_shift, max_shift:-max_shift]
        shape = hr.shape
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        ii = self.shift[0]
        jj = self.shift[1]
        ii_end = shape[0] - (2 * max_shift - ii)
        jj_end = shape[1] - (2 * max_shift - jj)
        registered_hr = hr[ii:ii_end, jj:jj_end]
        registered_mask = mask[ii:ii_end, jj:jj_end]
        cssim = cSSIM(sr_cropped, registered_hr, registered_mask)
        return cssim

    def get_clpips(self, sr, hr, mask=None, max_shift=3):
        hr = hr.astype(float)
        sr = sr.astype(float)
        sr_cropped = sr[max_shift:-max_shift, max_shift:-max_shift]
        shape = hr.shape
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        ii = self.shift[0]
        jj = self.shift[1]
        ii_end = shape[0] - (2 * max_shift - ii)
        jj_end = shape[1] - (2 * max_shift - jj)
        registered_hr = hr[ii:ii_end, jj:jj_end]
        registered_mask = mask[ii:ii_end, jj:jj_end]
        result = self.get_lpips(sr_cropped, registered_hr, registered_mask)
        return result




