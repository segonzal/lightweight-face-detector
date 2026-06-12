import os
import yaml
import numpy as np
from PIL import Image
from collections import namedtuple

from torch.utils.data import Dataset, DataLoader


Sample = namedtuple("Sample", ["image", "bbox", "kps"])


class WiderFaceRetinaDataset(Dataset):
    def __init__(self, img_dir, label_path, transform=None):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image folder not found: {img_dir}\n"
                f"Run download.sh to download the dataset."
            )
        if not os.path.isfile(label_path):
            raise FileNotFoundError(
                f"Labels file not found: {label_path}\n"
                f"Run download.sh to download the dataset."
            )
        
        self.img_dir = img_dir
        self.label_path = label_path
        self.transform = transform
        self.samples = self._load_retinaface_annotations(label_path)

    def _load_retinaface_annotations(self, path):
        samples = []

        with open(path, 'r') as fp:
            for line in fp:
                line = line.strip()
                if line.startswith('#'):
                    samples.append( Sample(line[2:], [], []) )
                else:
                    line = line.split(' ')
                    samples[-1].bbox.append(np.array(line[0:4], dtype=np.float32))
                    samples[-1].kps.append(np.array(line[4:19], dtype=np.float32).reshape(5, 3))
        
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = os.path.join(self.img_dir, sample.image)
        image = Image.open(img_path).convert("RGB")

        bboxes = np.stack(sample.bbox)
        kps = np.stack(sample.kps)

        if self.transform:
            image, bboxes, kps = self.transform(image, bboxes, kps)
    
        return image, {"bbox": bboxes, "kps": kps}


class PersonalFaceDataset(Dataset):
    def __init__(self, transform=None): ...
    def __len__(self): ...        
    def __getitem__(self, idx): ...


def load_dataset_paths(path="data/datasets.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataloader(loader_cfg, split, transform):
    dataset_type = loader_cfg['dataset']
    dataset_paths = load_dataset_paths()

    if split not in dataset_paths[dataset_type]:
        raise ValueError(f"Split '{split}' is undefined for dataset '{dataset_type}' in datasets.yaml")

    split_cfg = dataset_paths[dataset_type][split]

    if dataset_type == "wider_face":
        dataset = WiderFaceRetinaDataset(
            img_dir=split_cfg['img_dir'],
            label_path=split_cfg['label_path'],
            transform=transform)
    else:
        raise ValueError(f"Unkown dataset type: {dataset_type}")
        
    should_shuffle = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg['batch_size'],
        shuffle=should_shuffle,
        num_workers=loader_cfg['num_workers']
    )
    
    return dataloader