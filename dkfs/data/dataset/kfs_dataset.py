import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as transforms
from tqdm import tqdm
import statistics

from data.dataset.base_dataset import BaseDataset
from data.dataset.similarity import KFS_SIFT_AS_DISTANCE
from data.image_folder import make_dataset

import torch

def compute_kfs_batch(batch):
    """batch = lista (p_path, ref_path)"""
    kfs = KFS_SIFT_AS_DISTANCE()  # każdy worker dostaje własny obiekt
    results = []
    for p_path, ref_path in batch:
        ref_img_ = Image.open(ref_path).convert('RGB')
        ref_gray = np.asarray(ImageOps.grayscale(ref_img_).getdata()).reshape(ref_img_.size)

        p_img_ = Image.open(p_path).convert('RGB')
        p_gray = np.asarray(ImageOps.grayscale(p_img_).getdata()).reshape(p_img_.size)

        kfs_value = kfs.compute_similarity(ref_gray, p_gray, -1)
        if kfs_value is not None:
            results.append((p_path, ref_path, kfs_value))
    return results


def chunkify(lst, n):
    """Podział listy na n równych części"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


class PairwiseKFSRawDataset(BaseDataset):
    def initialize(self, dataroots, load_size=64, num_workers=5):
        if not isinstance(dataroots, list):
            dataroots = [dataroots, ]
        self.roots = dataroots
        self.load_size = load_size

        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.ref_paths = sorted(make_dataset(self.dir_ref))

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = sorted(make_dataset(self.dir_p0))

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = sorted(make_dataset(self.dir_p1))

        transform_list = [
            transforms.Resize(load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.transform = transforms.Compose(transform_list)

        # plik cache (np. dataset/cache.pkl)
        cache_file = os.path.join(self.roots[0], "cache_kfs_full.pkl")

        if os.path.exists(cache_file):
            print(f"[KFS] Loading cache from {cache_file}")
            with open(cache_file, "rb") as f:
                self.samples = pickle.load(f)
        else:
            print("[KFS] Precomputing similarities...")
            tasks = []
            for p0_path, p1_path, ref_path in zip(self.p0_paths, self.p1_paths, self.ref_paths):
                tasks.append((p0_path, ref_path))
                tasks.append((p1_path, ref_path))

            # dzielimy na ~num_workers kawałków
            batches = chunkify(tasks, num_workers)

            self.samples = []
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for res in tqdm(ex.map(compute_kfs_batch, batches), total=len(batches),
                                desc="Precomputing KFS (parallel batches)"):
                    self.samples.extend(res)

            # zapis cache
            with open(cache_file, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"[KFS] Cache saved to {cache_file}")

        # wyciągamy trzeci element z każdej tupli
        values = [t[2] for t in self.samples]

        # obliczamy statystyki
        minimum = min(values)
        maximum = max(values)
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)  # odchylenie standardowe

        print(f"Min: {minimum}, Max: {maximum}, Mean: {mean}, Std: {std_dev}")
        print(f"Samples count: {len(self.samples)}")
        self.samples = [(t[0], t[1], (t[2] - minimum) / (maximum - minimum)) for t in self.samples]

    def __getitem__(self, index):
        p_path, ref_path, kfs_value = self.samples[index]

        p_img_ = Image.open(p_path).convert('RGB')
        ref_img_ = Image.open(ref_path).convert('RGB')

        p_img = self.transform(p_img_)
        ref_img = self.transform(ref_img_)

        judge_kfs = torch.FloatTensor([kfs_value])

        return {
            'p': p_img,
            'ref': ref_img,
            'p_path': p_path,
            'ref_path': ref_path,
            'judge_kfs': judge_kfs
        }

    def __len__(self):
        return len(self.samples)