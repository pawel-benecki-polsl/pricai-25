import os.path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from data.dataset.base_dataset import BaseDataset
from data.dataset.similarity import KFS_SIFT_AS_DISTANCE
from data.image_folder import make_dataset


# from IPython import embed

machine_judges = dict()
machine_judges_raw = dict()
kfs = KFS_SIFT_AS_DISTANCE()


class TwoAFCDataset(BaseDataset):

    def initialize(self, dataroots, load_size=64):
        if not isinstance(dataroots, list):
            dataroots = [dataroots, ]
        self.roots = dataroots
        self.load_size = load_size

        # image directory
        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.ref_paths = sorted(make_dataset(self.dir_ref))

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = sorted(make_dataset(self.dir_p0))

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = sorted(make_dataset(self.dir_p1))

        transform_list = []
        transform_list.append(transforms.Resize(load_size))
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        self.human_judge_paths = sorted(make_dataset(self.dir_J, mode="np"))

    def __getitem__(self, index):
        global machine_judges
        global kfs
        p0_path = self.p0_paths[index]
        p1_path = self.p1_paths[index]
        ref_path = self.ref_paths[index]
        human_judge_path = self.human_judge_paths[index]

        p0_img_ = Image.open(p0_path).convert('RGB')
        p0_img = self.transform(p0_img_)

        p1_img_ = Image.open(p1_path).convert('RGB')
        p1_img = self.transform(p1_img_)

        ref_img_ = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img_)

        if index not in machine_judges:
            p0_gray = np.asarray(ImageOps.grayscale(p0_img_).getdata()).reshape(p0_img_.size)
            p1_gray = np.asarray(ImageOps.grayscale(p1_img_).getdata()).reshape(p1_img_.size)
            ref_gray = np.asarray(ImageOps.grayscale(ref_img_).getdata()).reshape(ref_img_.size)

            p0_d = kfs.compute_similarity(ref_gray, p0_gray, -1)
            p1_d = kfs.compute_similarity(ref_gray, p1_gray, -1)
            machine_judges_raw[index] = p0_d, p1_d
            if p0_d is None or p1_d is None:
                judge_kfs = 0.5
            else:
                judge_kfs = 0.0 if p0_d < p1_d else (1.0 if p1_d < p0_d else 0.5)
            machine_judges[index] = judge_kfs

        human_judge = np.load(human_judge_path)[0]
        machine_judge = machine_judges[index]
        d0 = machine_judges_raw[index][0]
        d1 = machine_judges_raw[index][1]

        # # Save images with specified naming convention
        # output_dir = r"e:\experiments\2afc_comparison2"
        # os.makedirs(output_dir, exist_ok=True)
        # prefix = "same" if human_judge == machine_judge else "different"
        # for img, suffix in [(p0_img_, "p0"), (p1_img_, "p1"), (ref_img_, "ref")]:
        #     filename = f"{prefix}_{index}_{suffix}_machine_{machine_judge}_human_{human_judge}_kfs0_{machine_judge_raw[0]}_kfs1_{machine_judge_raw[1]}.png"
        #     filepath = os.path.join(output_dir, filename)
        #     if not os.path.exists(filepath):
        #         img.save(filepath)

        judge = np.asarray((2 * human_judge + machine_judge) / 3)

        judge_img = judge.reshape((1, 1, 1,))  # [0,1]
        judge_img = torch.FloatTensor(judge_img)

        if d0 is None or d1 is None:
            return None

        return {'p0': p0_img, 'p1': p1_img, 'ref': ref_img, 'judge': judge_img, "d0": d0, "d1" : d1,
                'p0_path': p0_path, 'p1_path': p1_path, 'ref_path': ref_path}

    def __len__(self):
        return len(self.p0_paths)


