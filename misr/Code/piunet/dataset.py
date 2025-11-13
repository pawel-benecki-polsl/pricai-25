import numpy as np
import torch
import os

from utils import gen_sub


class ProbaVDatasetTrain(torch.utils.data.Dataset):
    """multitemporal scenes."""

    def __init__(self, config):
        x_LR = np.load(config.train_lr_file).astype(np.float32)
        x_HR = np.load(config.train_hr_file).astype(np.float32)
        M = np.load(config.train_masks_file).astype(np.float32)

        x_LR = x_LR[:config.max_train_scenes]
        x_HR = x_HR[:config.max_train_scenes]
        M = M[:config.max_train_scenes]

        self.x_LR_patches = gen_sub(x_LR, config.patch_size, config.patch_size)
        self.x_HR_patches = gen_sub(x_HR, config.patch_size * 3, config.patch_size * 3)
        self.M_patches = gen_sub(M, config.patch_size * 3, config.patch_size * 3)

        valid_pos = np.mean(self.M_patches, (1, 2, 3)) > 0.1  # extra clearance check on patches
        self.x_LR_patches = self.x_LR_patches[valid_pos]
        self.x_HR_patches = self.x_HR_patches[valid_pos]
        self.M_patches = self.M_patches[valid_pos]

        self.mu = 7433.6436
        self.sigma = 2353.0723

    def __len__(self):
        return len(self.x_LR_patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_LR = (self.x_LR_patches[idx] - self.mu) / self.sigma
        x_HR = self.x_HR_patches[idx]
        M = self.M_patches[idx]

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(M, (2, 0, 1)).astype(np.float32)  # (T,X,Y)


class ProbaVDatasetVal(torch.utils.data.Dataset):
    """multitemporal scenes."""

    def __init__(self, config):
        self.x_LR = np.load(config.val_lr_file).astype(np.float32)  # [:config.max_val_scenes]
        self.x_HR = np.load(config.val_hr_file).astype(np.float32)  # [:config.max_val_scenes]
        self.M = np.load(config.val_masks_file).astype(np.float32)  # [:config.max_val_scenes]

        self.mu = 7433.6436
        self.sigma = 2353.0723

    def __len__(self):
        return len(self.x_LR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_LR = (self.x_LR[idx] - self.mu) / self.sigma
        x_HR = (self.x_HR[idx] - self.mu) / self.sigma
        M = self.M[idx]

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(M, (2, 0, 1)).astype(np.float32)


def _loadz(path: str):
    res = np.load(path)
    return [res[f].astype(np.float32) for f in res.files]


class Mus2Config:
    X_mean = 2289.4213450227244
    X_std = 1178.364076456715
    y_mean = 49.29313061400437
    y_std = 26.912620163088683


class Mus2Dataset(torch.utils.data.Dataset):

    @staticmethod
    def sub_images(X, d):
        n_x = X.shape[0] // d
        n_y = X.shape[1] // d
        k = []

        for i in range(n_x):
            for j in range(n_y):
                sub = X[i * d:i * d + d, j * d:j * d + d]
                k.append(sub)
        return k

    @staticmethod
    def gen_sub(scenes, d):
        ch = np.min([im.shape[2] for im in scenes])

        X_sub = []
        for i, X in enumerate(scenes):
            sub = Mus2Dataset.sub_images(X[:, :, :ch], d)
            X_sub = X_sub + sub

        X_sub = np.stack(X_sub)
        print(X_sub.shape)
        return X_sub

    def __init__(self, config=None):
        patch_size = 40
        train_percent = 0.5
        val_percent = 0.1
        test_percent = 0.2

        split_file_path = f"C:\\projekty\\piunet\\Dataset\\mus2_split_{patch_size}.npz"
        x_LR = _loadz("C:\\projekty\\piunet\\Dataset\\mus2_x.npz")
        x_HR = _loadz("C:\\projekty\\piunet\\Dataset\\mus2_y.npz")

        x_LR_patches_all = Mus2Dataset.gen_sub(x_LR, patch_size)
        x_HR_patches_all = Mus2Dataset.gen_sub(x_HR, patch_size * 3)

        self.x_LR_patches = x_LR_patches_all[::3, ...]
        self.x_HR_patches = x_HR_patches_all[::3, ...]

        if not os.path.isfile(split_file_path):
            idx = np.random.rand(int(self.x_LR_patches.shape[0]))
            train_indices = np.argwhere(idx < train_percent)
            val_indices = np.argwhere((idx >= train_percent) & (idx < (train_percent + val_percent)))
            test_indices = np.argwhere(idx >= (1.0 - test_percent))
            np.savez(split_file_path, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

        split_file = np.load(split_file_path)
        self.train_indices = split_file["train_indices"].flatten()
        self.val_indices = split_file["val_indices"].flatten()
        self.test_indices = split_file["test_indices"].flatten()

        print("Loaded the dataset")


class Mus2Dataset200pxLR(torch.utils.data.Dataset):

    @staticmethod
    def sub_images(X, d):
        n_x = X.shape[0] // d
        n_y = X.shape[1] // d
        k = []

        for i in range(n_x):
            for j in range(n_y):
                sub = X[i * d:i * d + d, j * d:j * d + d]
                k.append(sub)
        return k

    @staticmethod
    def gen_sub(scenes, d):
        ch = np.min([im.shape[2] for im in scenes])

        X_sub = []
        for i, X in enumerate(scenes):
            sub = Mus2Dataset.sub_images(X[:, :, :ch], d)
            X_sub = X_sub + sub

        X_sub = np.stack(X_sub)
        print(X_sub.shape)
        return X_sub

    def __init__(self, config=None):
        patch_size = 200

        x_LR = _loadz("C:\\projekty\\piunet\\Dataset\\mus2_x.npz")
        x_HR = _loadz("C:\\projekty\\piunet\\Dataset\\mus2_y.npz")

        self.x_LR_patches = Mus2Dataset.gen_sub(x_LR, patch_size)
        self.x_HR_patches = Mus2Dataset.gen_sub(x_HR, patch_size * 3)

        print("Loaded the dataset")

    def __len__(self):
        return self.x_HR_patches.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_LR = (self.x_LR_patches[idx] - Mus2Config.X_mean) / Mus2Config.X_std
        x_HR = (self.x_HR_patches[idx] - Mus2Config.y_mean) / Mus2Config.y_std

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(np.ones_like(x_HR), (2, 0, 1)).astype(np.float32)  # (T,X,Y)


# if __name__ == "__main__":
#     d = Mus2Dataset()


class Mus2DatasetTrain(Mus2Dataset):

    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.train_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dest_idx = self.train_indices[idx]

        x_LR = (self.x_LR_patches[dest_idx] - Mus2Config.X_mean) / Mus2Config.X_std
        x_HR = (self.x_HR_patches[dest_idx] - Mus2Config.y_mean) / Mus2Config.y_std

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(np.ones_like(x_HR), (2, 0, 1)).astype(np.float32)  # (T,X,Y)


class Mus2DatasetVal(Mus2Dataset):

    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.val_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dest_idx = self.train_indices[idx]

        x_LR = (self.x_LR_patches[dest_idx] - Mus2Config.X_mean) / Mus2Config.X_std
        x_HR = (self.x_HR_patches[dest_idx] - Mus2Config.y_mean) / Mus2Config.y_std

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(np.ones_like(x_HR), (2, 0, 1)).astype(np.float32)


class Mus2DatasetTest(Mus2Dataset):

    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.test_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dest_idx = self.test_indices[idx]

        x_LR = (self.x_LR_patches[dest_idx] - Mus2Config.X_mean) / Mus2Config.X_std
        x_HR = (self.x_HR_patches[dest_idx] - Mus2Config.y_mean) / Mus2Config.y_std

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(np.ones_like(x_HR), (2, 0, 1)).astype(np.float32)


####################################################
class HarvardConfig:
    X_mean = 3394.8581659973224
    X_std = 1776.0605839614523
    y_mean = 12.730813146766963
    y_std = 6.580535330796187


class HarvardDataset(torch.utils.data.Dataset):

    @staticmethod
    def sub_images(X, d):
        n_x = X.shape[0] // d
        n_y = X.shape[1] // d
        k = []

        for i in range(n_x):
            for j in range(n_y):
                sub = X[i * d:i * d + d, j * d:j * d + d]
                k.append(sub)
        return k

    @staticmethod
    def gen_sub(scenes, d):
        ch = np.min([im.shape[2] for im in scenes])

        X_sub = []
        for i, X in enumerate(scenes):
            sub = Mus2Dataset.sub_images(X[:, :, :ch], d)
            X_sub = X_sub + sub

        X_sub = np.stack(X_sub)
        print(X_sub.shape)
        return X_sub

    def __init__(self, config=None):
        patch_size = 40
        train_percent = 0.5
        val_percent = 0.1
        test_percent = 0.2

        split_file_path = f"c:\\datasets\\harvard_s2\\mus2_split_{patch_size}.npz"
        x_LR = _loadz("c:\\datasets\\harvard_s2\\mus2_x.npz")
        x_HR = _loadz("c:\\datasets\\harvard_s2\\mus2_y.npz")

        self.x_LR_patches = np.stack(x_LR)
        self.x_HR_patches = np.stack(x_HR)

        if not os.path.isfile(split_file_path):
            idx = np.random.rand(int(self.x_LR_patches.shape[0]))
            train_indices = np.argwhere(idx < train_percent)
            val_indices = np.argwhere((idx >= train_percent) & (idx < (train_percent + val_percent)))
            test_indices = np.argwhere(idx >= (1.0 - test_percent))
            np.savez(split_file_path, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

        split_file = np.load(split_file_path)
        self.train_indices = split_file["train_indices"].flatten()
        self.val_indices = split_file["val_indices"].flatten()
        self.test_indices = split_file["test_indices"].flatten()

        print("Loaded the dataset")


class HarvardDatasetTest(HarvardDataset):

    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.x_HR_patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_LR = (self.x_LR_patches[idx] - HarvardConfig.X_mean) / HarvardConfig.X_std
        x_HR = (self.x_HR_patches[idx] - HarvardConfig.y_mean) / HarvardConfig.y_std

        return np.transpose(x_LR, (2, 0, 1)).astype(np.float32), np.transpose(x_HR, (2, 0, 1)).astype(
            np.float32), np.transpose(np.ones_like(x_HR), (2, 0, 1)).astype(np.float32)


if __name__ == "__main__":
    ds = Mus2Dataset()
