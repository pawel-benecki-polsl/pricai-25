import os

import torch.utils.data

from data.base_data_loader import BaseDataLoader


def CreateDataset(dataroots,dataset_mode='2afc',load_size=64,):
    dataset = None
    if dataset_mode=='2afc': # human judgements
        from data.dataset.twoafc_dataset import TwoAFCDataset
        dataset = TwoAFCDataset()
    elif dataset_mode=='jnd': # human judgements
        from data.dataset.jnd_dataset import JNDDataset
        dataset = JNDDataset()
    else:
        raise ValueError("Dataset Mode [%s] not recognized."%dataset_mode)

    dataset.initialize(dataroots,load_size=load_size)
    return dataset

def collate_skip_none(batch):
    # filter out None items
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # optional: skip if entire batch is invalid
    return torch.utils.data.default_collate(batch)


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, datafolders, dataroot='./dataset',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
        BaseDataLoader.initialize(self)
        if(not isinstance(datafolders,list)):
            datafolders = [datafolders,]
        dataroot_abspath = os.path.abspath(dataroot)
        data_root_folders = [os.path.join(dataroot_abspath,datafolder) for datafolder in datafolders]
        self.dataset = CreateDataset(data_root_folders,dataset_mode=dataset_mode,load_size=load_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(nThreads),
            collate_fn=collate_skip_none)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
