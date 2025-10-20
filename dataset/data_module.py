from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from dataset.data_helper_sn import create_datasets_sn
from dataset.data_helper_mn import create_datasets_mn
from config.config import parser
import torch
import numpy as np
from torch.utils.data.sampler import RandomSampler
import math
import random



class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        if  self.args.test_mode =='train_2':
            train_mn, dev_mn, test_mn = create_datasets_mn(self.args)
            self.dataset = {
                "train": train_mn, "validation": dev_mn, "test": test_mn
            }
        if self.args.test_mode =='sn':
            train_sn, dev_sn, test_sn = create_datasets_sn(self.args)
            self.dataset = {
                "train": test_sn, "validation": test_sn, "test": test_sn
            }
        if self.args.test_mode =='mn':
            train_mn, dev_mn, test_mn = create_datasets_mn(self.args)
            self.dataset = {
                "train": test_mn, "validation": test_mn, "test": test_mn
            }

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        if self.args.test_mode == 'train_2':
            loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,shuffle=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader

        else:
            loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,shuffle=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader

    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        if self.args.test_mode == 'train_2':
            loader = DataLoader(self.dataset["validation"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,
                                shuffle=False,
                                num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader
        else:
            loader = DataLoader(self.dataset["validation"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,
                                shuffle=False,
                                num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader



