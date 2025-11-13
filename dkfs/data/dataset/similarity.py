"""Module that stores similarity metric routines."""
import abc
import os

import sewar

if os.name != 'nt':
    raise NotImplementedError("This module will run only on Windows.")

import numpy as np
import cv2
import itertools
import lpips
import torch

from scipy.ndimage import gaussian_filter, uniform_filter


class Metric2(abc.ABC):
    @abc.abstractmethod
    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        pass


def _to_8bit_min_max(img: np.ndarray) -> np.ndarray:
    _min = np.min(img)
    _max = np.max(img)
    _range = _max - _min
    if _range == 0:
        _scaled = np.zeros_like(img, dtype=np.uint8)
    else:
        _scaled = (255.0 * (img.astype(np.float32) - _min) / _range).astype(np.uint8)
    return _scaled


class KFS_SIFT_AS_DISTANCE(Metric2):
    FEATURES = cv2.SIFT_create()

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float = None, mask: np.ndarray = None) -> float:
        ref8b = _to_8bit_min_max(gt)
        img8b = _to_8bit_min_max(img)

        kps = list(KFS_SIFT_AS_DISTANCE.FEATURES.detect(ref8b, None))  # No mask, used only on non-masked images
        # kps += list(KFS_SIFT_AS_DISTANCE.FEATURES.detect(img8b, None))  # No mask, used only on non-masked images

        if len(kps) < 5:
            return None

        _, ref_descr = KFS_SIFT_AS_DISTANCE.FEATURES.compute(ref8b, kps)
        _, img_descr = KFS_SIFT_AS_DISTANCE.FEATURES.compute(img8b, kps)

        kp_count = len(kps)

        desc_diffs = (ref_descr - img_descr) / 200
        distance = 0
        for desc_diff in desc_diffs:
            distance += np.linalg.norm(desc_diff)

        return distance / kp_count


class KFS_SIFT(Metric2):
    distance_mteric = KFS_SIFT_AS_DISTANCE()

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        return 1000 / KFS_SIFT.distance_mteric.compute_similarity(gt, img, max_value, mask)


class KFS_SURF(Metric2):

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        raise NotImplementedError("No SURF in Python cv2")


class CPSNR(Metric2):
    @staticmethod
    def cPSNR(sr: np.ndarray, hr: np.ndarray, hr_map: np.ndarray, max_value: float) -> float:
        """
        Clear Peak Signal-to-Noise Ratio. The PSNR score, adjusted for brightness and other volatile features, e.g. clouds.
        Args:
            sr: numpy.ndarray (n, m), super-resolved image
            hr: numpy.ndarray (n, m), high-res ground-truth image
            hr_map: numpy.ndarray (n, m), status map of high-res image, indicating clear pixels by a value of 1
        Returns:
            cPSNR: float, score
        """

        if len(sr.shape) == 2:
            sr = sr[None,]
            hr = hr[None,]
            hr_map = hr_map[None,]

        _sr = sr.astype(np.float32) / max_value
        _hr = hr.astype(np.float32) / max_value

        n_clear = np.sum(hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
        if np.any(n_clear == 0):
            return np.nan
        diff = _hr - _sr
        bias = np.sum(diff * hr_map, axis=(1, 2)) / n_clear  # brightness bias
        cMSE = np.sum(np.square((diff - bias[:, None, None]) * hr_map), axis=(1, 2)) / n_clear
        cPSNR = -10 * np.log10(cMSE)  # + 1e-10)

        if cPSNR.shape[0] == 1:
            cPSNR = cPSNR[0]

        return cPSNR

    @staticmethod
    def get_patch(img: np.ndarray, x: int, y: int, size: int):
        """
        Slices out a square patch from `img` starting from the (x,y) top-left corner.
        If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
        and the output has shape (l, size, size).
        Args:
            img: numpy.ndarray (n, m), input image
            x, y: int, top-left corner of the patch
            size: int, patch size
        Returns:
            patch: numpy.ndarray (size, size)
        """

        patch = img[..., x:(x + size), y:(y + size)]  # using ellipsis to slice arbitrary ndarrays
        return patch

    @staticmethod
    def patch_iterator(img, positions, size):
        """Iterator across square patches of `img` located in `positions`."""
        for x, y in positions:
            yield CPSNR.get_patch(img=img, x=x, y=y, size=size)

    @staticmethod
    def shift_cPSNR(sr, hr, hr_map, max_value, border_w=3):
        """
        cPSNR score adjusted for registration errors. Computes the max cPSNR score across shifts of up to `border_w` pixels.
        Args:
            sr: np.ndarray (n, m), super-resolved image
            hr: np.ndarray (n, m), high-res ground-truth image
            hr_map: np.ndarray (n, m), high-res status map
            border_w: int, width of the trimming border around `hr` and `hr_map`
        Returns:
            max_cPSNR: float, score of the super-resolved image
        """

        size = sr.shape[1] - (2 * border_w)  # patch size
        sr = CPSNR.get_patch(img=sr, x=border_w, y=border_w, size=size)
        pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
        iter_hr = CPSNR.patch_iterator(img=hr, positions=pos, size=size)
        iter_hr_map = CPSNR.patch_iterator(img=hr_map, positions=pos, size=size)
        site_cPSNR = np.array([CPSNR.cPSNR(sr, hr, hr_map, max_value) for hr, hr_map in zip(iter_hr, iter_hr_map)])
        max_cPSNR = np.max(site_cPSNR, axis=0)
        return max_cPSNR

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        return CPSNR.shift_cPSNR(img, gt, mask, max_value)


class IFC(Metric2):
    # TODO
    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        raise NotImplementedError("Not implemented")


class PSNR(Metric2):
    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        return sewar.psnr(gt, img, MAX=max_value)


class PSNR_HF(Metric2):
    SIGMA = 1.5
    _PSNR = PSNR()

    def _blur(self, img: np.ndarray) -> np.ndarray:
        return gaussian_filter(img, PSNR_HF.SIGMA)

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        gt_dog = self._blur(np.abs(gt - self._blur(gt)))
        img_dog = self._blur(np.abs(img - self._blur(img)))
        return PSNR_HF._PSNR.compute_similarity(gt_dog, img_dog, max_value)


class LPIPS(Metric2):
    def normalize_m1_1(self, img: np.ndarray):
        _min, _max = np.min(img), np.max(img)
        width = _max - _min
        res = img
        if width != 0.0:
            res = (2 * (img - _min) / width) - 1.0
        return res

    def __init__(self):
        self.met = lpips.LPIPS(net='alex')

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None) -> float:
        # 1, 3, d d
        gt_ = self.normalize_m1_1(gt).astype(np.float32)
        img_ = self.normalize_m1_1(img).astype(np.float32)
        gt_ = torch.from_numpy(np.expand_dims(np.stack([gt_, gt_, gt_]), 0))
        img_ = torch.from_numpy(np.expand_dims(np.stack([img_, img_, img_]), 0))
        result = self.met.forward(gt_, img_)
        return -result.detach().cpu().numpy().flatten()[0]


class PSNR_LSD(Metric2):
    SIGMA = 1.5
    STD_WIN_SIZE = 3
    _PSNR = PSNR()

    @staticmethod
    def _blur(img: np.ndarray) -> np.ndarray:
        return gaussian_filter(img, PSNR_LSD.SIGMA)

    @staticmethod
    def _stdfilter(img: np.ndarray) -> np.ndarray:
        _mean = uniform_filter(img, (PSNR_LSD.STD_WIN_SIZE, PSNR_LSD.STD_WIN_SIZE))
        _sqr_mean = uniform_filter(img ** 2, (PSNR_LSD.STD_WIN_SIZE, PSNR_LSD.STD_WIN_SIZE))
        return np.sqrt(_sqr_mean - _mean ** 2)

    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None):
        return PSNR_LSD._PSNR.compute_similarity(self._blur(self._stdfilter(gt)),
                                                 self._blur(self._stdfilter(img)),
                                                 max_value)


class SSIM(Metric2):
    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None):
        _ssim, _cs = sewar.ssim(gt, img, MAX=max_value)
        return _ssim


class UIQI(Metric2):
    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None):
        return sewar.uqi(gt, img)


class VIFp(Metric2):
    def compute_similarity(self, gt: np.ndarray, img: np.ndarray, max_value: float, mask: np.ndarray = None):
        return sewar.vifp(gt, img)
