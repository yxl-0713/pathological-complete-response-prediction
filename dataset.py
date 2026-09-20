import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchio as tio

class MRIDataset(Dataset):
    def __init__(self, timepoint_dirs, label_dict, valid_ids, input_mode="delta", augment=False):
        self.dirs = timepoint_dirs
        self.labels = label_dict
        self.valid_ids = valid_ids
        self.augment = augment
        self.input_mode = input_mode
        self.files = self.valid_ids
        self.transforms = tio.Compose([
            tio.RandomAffine(scales=0.05, degrees=5, translation=3, p=0.3),
            tio.RandomNoise(mean=0, std=0.01, p=0.2),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        files_list = list(self.files)
        fname = files_list[idx]
        label = float(self.labels[fname])
        vols = []

        for i, path in enumerate(self.dirs):
            nii = nib.load(os.path.join(path, f"{fname}.nii.gz"))
            vol = nii.get_fdata()
            p2, p98 = np.percentile(vol, (2, 98))
            vol = np.clip(vol, p2, p98)
            vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-5)
            vol = F.interpolate(torch.tensor(vol).float()[None, None], size=(64,64,64), mode='trilinear', align_corners=False)
            vols.append(vol.squeeze())

        t0, t1 = vols
        if self.input_mode == "delta":
            x = torch.stack([t0, t1], dim=0)
        if self.augment:
            x = self.transforms(tio.Image(tensor=x, type=tio.INTENSITY)).data
        return x, torch.tensor(label, dtype=torch.float32)
