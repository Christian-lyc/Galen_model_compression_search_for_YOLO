import abc
from abc import abstractmethod

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms
from torch.utils.data import SubsetRandomSampler, Subset

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import LOGGER


class ADataProvider(metaclass=abc.ABCMeta):

    @property
    @abstractmethod
    def train_loader(self):
        pass

    @property
    @abstractmethod
    def val_loader(self):
        pass

    @property
    @abstractmethod
    def test_loader(self):
        pass

    @property
    @abstractmethod
    def sens_loader(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def batch_input_shape(self):
        pass

    @abstractmethod
    def get_random_tensor_with_input_shape(self):
        pass


class CIFAR10Provider(ADataProvider):

    def __init__(self,
                 target_device: torch.device,
                 data_dir="./data",
                 batch_size=256,
                 sensitivity_sample_count=128,
                 seed=42,
                 num_workers=16,
                 split_ratio=0.2,
                 **kwargs
                 ):
        self._batch_size = batch_size
        self._target_device = target_device
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        base_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )
        self._val_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=base_transform, download=True)
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )
        self._train_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_transform)
        self._test_set = torchvision.datasets.CIFAR10(data_dir, train=False, transform=base_transform)
        num_images = len(self._train_set)
        indices = list(range(num_images))
        split_idx = int(np.floor(num_images * split_ratio))
        random_gen = np.random.Generator(np.random.PCG64(seed))
        random_gen.shuffle(indices)
        train_idx, val_idx = indices[split_idx:], indices[:split_idx]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        # although the val_set is used the sens_sample ensures that train images are used for sensitivity analysis
        # train_set could not be used due to random augmentation
        sens_set = Subset(self._val_set, train_idx[:sensitivity_sample_count])

        self._train_loader = torch.utils.data.DataLoader(self._train_set, batch_size=batch_size, sampler=train_sampler,
                                                         num_workers=num_workers, pin_memory=True)
        self._valid_loader = torch.utils.data.DataLoader(self._val_set, batch_size=batch_size, sampler=valid_sampler,
                                                         num_workers=num_workers, pin_memory=True)
        self._sens_loader = torch.utils.data.DataLoader(sens_set, batch_size=batch_size, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(self._test_set, batch_size=batch_size, shuffle=True,
                                                        pin_memory=True)

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._valid_loader

    @property
    def sens_loader(self):
        return self._sens_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def num_classes(self):
        return 10

    @property
    def batch_input_shape(self):
        # return next(iter(self.train_loader))[0].shape # causes a memory leak
        return torch.Size([self._batch_size, 3, 32, 32])

    def get_random_tensor_with_input_shape(self):
        return torch.randn(self.batch_input_shape).to(self._target_device)


class YOLODetectionProvider(ADataProvider):
    def __init__(self,
                 target_device: torch.device,
                 data_dir="data.yaml",
                 imgsz=640,
                 batch_size=16,
                 sensitivity_sample_count=128,
                 seed=42,
                 num_workers=4,
                 classes=6,
                 model=None,
                 split_ratio=0.2,
                 **kwargs):
        self._batch_size = batch_size
        self._target_device = target_device
        self.model = model



        # Load YOLO-style dataset config
        data_dict = check_det_dataset(data_dir)
        trainset=data_dict["train"]
        valset=data_dict.get("val")
        testset=data_dict.get("test")
        self._num_classes = int(data_dict.get('nc', classes))

        # Load default YOLO config and override necessary fields
        cfg = get_cfg(DEFAULT_CFG,self.model.overrides)

        for key, value in {
        'task': 'detect',
        'data': data_dir,
        'imgsz': imgsz,
        'batch': batch_size,
        'workers': num_workers,
        'seed': seed

        }.items():
            setattr(cfg, key, value)
        self.data = check_det_dataset(cfg.data)
        self.cfg = cfg
        self.cfg.mosaic=0

        #parent class has init function, children class don't, then fill the args for parent class
        self._train_loader=self.train_get_dataloader(trainset,batch_size=batch_size,rank=-1,mode="train")

        self._val_loader=self.get_dataloader(valset, batch_size)

        # Extract a sensitivity dataset (subset of validation)
        # val_dataset_obj = self._val_loader.dataset
        # indices = list(range(len(self._val_loader.dataset)))
        # random_gen = np.random.Generator(np.random.PCG64(seed))
        # random_gen.shuffle(indices)
        ## sens_indices = list(range(min(sensitivity_sample_count, len(train_dataset))))
        # indices = indices[:sensitivity_sample_count]
        # sens_subset = torch.utils.data.Subset(val_dataset_obj, indices)
        #
        # self._sens_loader = torch.utils.data.DataLoader(
        #     sens_subset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=num_workers
        # )

        self._sens_loader=self._val_loader
        # Set test loader to val loader
        self._test_loader=self.get_dataloader(testset, batch_size)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """

        return build_yolo_dataset(self.cfg, img_path, batch, self.data, mode=mode, stride=32)
    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.cfg.workers, shuffle=False, rank=-1)

    def train_get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.train_build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.cfg.workers if mode == "train" else self.cfg.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def train_build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.cfg, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def sens_loader(self):
        return self._sens_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def batch_input_shape(self):
        return torch.Size([self._batch_size, 3, 640, 640])

    def get_random_tensor_with_input_shape(self):
        return torch.randn(self.batch_input_shape).to(self._target_device)
