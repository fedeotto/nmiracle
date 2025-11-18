import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from nmiracle.data.datasets import AlbertsDataset,PreTrainDataset
from nmiracle.utils.utils import seed_worker
import numpy as np
import os
from pathlib import Path
import pickle
import h5py

class SpectralDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.tokenizer = tokenizer
        self.training_stage = config.training_stage
        self.seed = config.seed
        
        # Set up data splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing"""
        if self.training_stage == "sub2struct":
            print(f"Setting pretrain dataset for substructure-to-structure training...")
            self._setup_pretrain_dataset(stage=stage)
        
        elif self.training_stage == "spec2struct":
            print("Setting up Alberts dataset for spectra-to-structure training...")
            self._setup_alberts_dataset(stage=stage)
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")

    def _setup_pretrain_dataset(self, stage=None):
        # Get paths from config
        data_dir = self.config.data_dir

        # Check for existing indices
        train_indices_path = Path(data_dir) / "pretrain_train_indices.npy"
        val_indices_path = Path(data_dir) / "pretrain_val_indices.npy"
        test_indices_path = Path(data_dir) / "pretrain_test_indices.npy"

        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)
        test_indices = np.load(test_indices_path)

        # Common dataset arguments
        common_args = {
            "data_dir": data_dir,
            "tokenizer": self.tokenizer,
            "max_molecule_len": self.config.max_molecule_len, 
            "max_substructures_len": self.config.max_substructures_len,
            "max_substructures_count": self.config.max_substructures_count
        }

        if stage == "fit" or stage is None:
            self.train_dataset = PreTrainDataset(indices=train_indices, **common_args)
            self.val_dataset = PreTrainDataset(indices=val_indices, **common_args)

        if stage == "test" or stage is None:
            self.test_dataset = PreTrainDataset(indices=test_indices, **common_args)

    def _setup_alberts_dataset(self, stage=None):
        """Set up datasets for spectra-to-all training"""        

        data_dir = self.config.data_dir

        # Load split indices (move logic from Dataset to DataModule)
        splits_path = os.path.join(data_dir, "split_indices.p")
        splits = np.load(splits_path, allow_pickle=True)
        
        train_indices = splits['train']
        val_indices = splits['val']
        test_indices = splits['test']

        common_args = {
            'data_dir': data_dir,
            'tokenizer': self.tokenizer,
            'n_ir_features': self.config.n_ir_features,
            'n_hnmr_features': self.config.n_hnmr_features,
            'n_cnmr_features': self.config.n_cnmr_features,
            'max_molecule_len': self.config.max_molecule_len,
            'use_ir': self.config.use_ir,
            'use_hnmr': self.config.use_hnmr,
            'use_cnmr': self.config.use_cnmr,
            'cnmr_binary': self.config.cnmr_binary,
            'cnmr_binary_bins': self.config.cnmr_binary_bins,
        }
        
        if stage == 'fit' or stage is None:
            self.train_dataset = AlbertsDataset(indices=train_indices, **common_args)
            self.val_dataset = AlbertsDataset(indices=val_indices, **common_args)

        if stage == 'test' or stage is None:
            self.test_dataset = AlbertsDataset(indices=test_indices, **common_args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            shuffle=True,
            num_workers=self.num_workers.train,
            prefetch_factor=self.prefetch_factor.train,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=seed_worker
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size.val,
            shuffle=False,
            num_workers=self.num_workers.val,
            prefetch_factor=self.prefetch_factor.val,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=seed_worker
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size.test,
            shuffle=False,
            num_workers=self.num_workers.test,
            prefetch_factor=self.prefetch_factor.test,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=seed_worker
            )