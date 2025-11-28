import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from common.utils import flatten_dict
from nmiracle.data.datamodule import SpectralDataModule
from nmiracle.utils.utils import get_wandb_name
from nmiracle.data.tokenizer import BasicSmilesTokenizer
from nmiracle.data.alphabet import generate_or_load_alphabet
import rootutils
import warnings
warnings.filterwarnings("ignore")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(config_path="../configs", config_name="nmiracle_config.yaml")
def main(config: DictConfig):
    assert config.training_stage in ['sub2struct', 'spec2struct'], \
        f"Unsupported training stage: {config.training_stage}"
    
    # Check pairing: spec2struct requires alberts_dataset
    if config.training_stage == 'spec2struct':
        assert config.data.dataset_name == 'alberts_dataset', \
            "Training stage 'spec2struct' requires 'alberts_dataset'"
    
    # Check pairing: sub2struct requires pretrain_dataset
    if config.training_stage == 'sub2struct':
        assert config.data.dataset_name == 'pretrain_dataset', \
            "Training stage 'sub2struct' requires 'pretrain_dataset'"

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)
    
    # Get current working directory from Hydra
    cwd = Path(os.getcwd())

    # Setup logs directory for this stage
    logs_dir = cwd / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    #Setting up the tokenizer
    tokenizer = BasicSmilesTokenizer()

    if config.data.alphabet is None:
        config.data.alphabet = Path(os.environ['PROJECT_ROOT'] / 'common' / 'alphabet.npy')

    alphabet  = generate_or_load_alphabet(config,
                                          alphabet_path=config.data.alphabet,
                                          tokenizer=tokenizer)
    tokenizer.setup_alphabet(alphabet)

    # Initialize data module
    data_module = SpectralDataModule(config=config.data, 
                                     tokenizer=tokenizer)
    # Setting up loggers
    loggers = []
    
    # Always add CSV logger
    csv_logger = pl.loggers.CSVLogger(
        save_dir=logs_dir, 
        name="",  
        version=""
    )
    loggers.append(csv_logger)
    
    # Add wandb logger if enabled
    if config.logger.wandb.enabled:
        # wandb_name = f"{config.logger.wandb.name}"
        wandb_name = get_wandb_name(config)
        wandb_logger = pl.loggers.WandbLogger(
            name=wandb_name,
            project=config.logger.wandb.project,
            entity=config.logger.wandb.entity,
            config=flatten_dict(OmegaConf.to_container(config, resolve=True)),
            save_dir=str(logs_dir / "wandb_logs"))
        loggers.append(wandb_logger)
    
    # Initialize model
    if config.training_stage == 'sub2struct':
        from nmiracle.models.pl_module import Sub2StructModule
        model = Sub2StructModule(config, tokenizer)
    elif config.training_stage == 'spec2struct':
        from nmiracle.models.pl_module import Spec2StructModule
        model = Spec2StructModule(config, tokenizer)
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cwd / "checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        monitor=config.callbacks.model_checkpoint.monitor,
        mode=config.callbacks.model_checkpoint.mode,
        save_last=config.callbacks.model_checkpoint.save_last)
    
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
        verbose=True)
    
    callbacks.append(early_stopping_callback)

    learning_rate_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(learning_rate_monitor)

    # Check for checkpoint resumption
    resume_from_checkpoint = None
    if hasattr(config, 'resume') and config.resume.checkpoint_path:
        resume_path = config.resume.checkpoint_path
        if os.path.exists(resume_path):
            resume_from_checkpoint = resume_path
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Warning: Checkpoint path {resume_path} not found, starting fresh training.")
    
    # Determine available devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    # elif torch.backends.mps.is_available():
    #     accelerator = "mps"
    #     devices = 1
    else:
        accelerator = "cpu"
        devices = 1  # For CPU training, use 1 device

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs, 
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        precision="16-mixed",
        log_every_n_steps=config.trainer.log_every_n_steps,
        strategy= DDPStrategy(find_unused_parameters=False) if devices > 1 else "auto",
        logger=loggers,
        inference_mode=False,
        callbacks=callbacks)
    
    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

if __name__ == "__main__":
    main()