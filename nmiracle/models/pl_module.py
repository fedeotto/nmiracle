import torch
import pytorch_lightning as pl
from nmiracle.models.sub2structmodel import Sub2Structure
from nmiracle.models.spec2structmodel import Spectra2Structure, MultiTaskSpectra2Structure
import gc
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class BaseModule(pl.LightningModule):
    "Base Pytorch lightning module"
    def __init__(self, config, tokenizer=None):
        super().__init__()
        #Save hyperparameters
        self.save_hyperparameters(config, ignore=['tokenizer'])
        self.config = config
        self.tokenizer = tokenizer
    
    def forward(self, batch):
        raise NotImplementedError("Forward method not implemented!")

    def on_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_start(self):
        """Evaluate generation on a single validation batch at the end of each training epoch (rank 0 only)."""
        if not self.trainer.is_global_zero:
            return
        
        self.eval()
        try:
            # Get a single batch from validation dataloader
            val_dataloader = self.trainer.datamodule.val_dataloader()
            val_batch = next(iter(val_dataloader))
            
            # Move batch to the correct device
            val_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in val_batch.items()}
            
            # Forward pass to get structure features
            with torch.no_grad():
                outputs = self.model(val_batch)
            
            # Get max molecule length from dataset
            max_molecule_len = self.trainer.datamodule.val_dataset.max_molecule_len
            
            # Evaluate generation
            generation_metrics = self.model.evaluate_generation(
                structure_features=outputs['structure_features'],
                structure_features_mask=outputs['structure_features_mask'],
                true_smiles=val_batch['molecules'],
                max_molecule_len=max_molecule_len,
                max_samples=None,
            )

            # Log metrics (no sync_dist needed since only rank 0 runs this)
            batch_size = len(val_batch['molecules'])
            for name, value in generation_metrics.items():
                self.log(f'val_generation/{name}', value, on_step=False, on_epoch=True, 
                         sync_dist=False, batch_size=batch_size, rank_zero_only=True)
                        
        except Exception as e:
            print(f"Warning: Generation evaluation failed: {e}")
        
        finally:
            self.train()

class Sub2StructModule(BaseModule):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.model = Sub2Structure(
            tokenizer=self.tokenizer,
            structure_kwargs=config.model.structure_model,
            pretrained_structure_path=config.model.pretrained_structure_model_path,
            num_substructures=config.data.num_substructures,
            max_substructures_count=config.data.dataset.max_substructures_count,
        )

    def training_step(self, batch, batch_idx):
        batch_size = len(batch['molecules'])
        outputs = self.model(batch)
        # Get loss
        loss, loss_info = self.model.compute_loss(outputs, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch['molecules'])
        outputs = self.model(batch)

        # Get loss
        loss, loss_info = self.model.compute_loss(outputs, batch)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        if 'structure_loss' in loss_info:
            self.log('val_structure_loss', loss_info['structure_loss'], prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def get_optimizer_params(self):
        return {
            'lr': 1.0e-5,
            'betas': (0.9, 0.98),
            'eps': 1.0e-9,
            'weight_decay': 1.0e-5
        }

    def configure_optimizers(self):
        # Create optimizer
        opt_params = self.get_optimizer_params()
        optimizer = torch.optim.Adam(
            self.parameters(),
            **opt_params
        )            
        return {
                'optimizer': optimizer
            }

class Spec2StructModule(BaseModule):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if config.model.use_multitask:
            self.model = MultiTaskSpectra2Structure(
                tokenizer=tokenizer,
                pretrained_structure_path=config.model.pretrained_structure_model_path,
                multispectra_encoder_kwargs=config.model.multispectra_encoder,
                structure_model_kwargs=config.model.structure_model,
                num_substructures=config.data.num_substructures,
                max_substructures_count = config.data.dataset.max_substructures_count,
            )
        else:
            self.model = Spectra2Structure(
                tokenizer=tokenizer,
                pretrained_spectrum_encoder_path= config.model.pretrained_spectrum_encoder_path,
                pretrained_structure_path=config.model.pretrained_structure_model_path,
                freeze_spectrum_encoder=False,
                multispectra_encoder_kwargs=config.model.multispectra_encoder,
                structure_model_kwargs=config.model.structure_model,
            )
        
    def validation_step(self, batch, batch_idx):
        # Get loss for spectra-to-structure stage
        batch_size = len(batch['molecules'])

        with torch.enable_grad(): #utilized in previous work due to some issues with nn.Transformer
            outputs = self.model(batch)

        loss, loss_info = self.model.compute_loss(outputs, batch)
        
        # Log
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        for name, value in loss_info.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # If multitask calculate and log substructure-prediction metrics
        if hasattr(self.model, 'calculate_substructure_metrics'):
            metrics = self.model.calculate_substructure_metrics(outputs=outputs, batch=batch)
            for name, value in metrics.items():
                self.log(f'val_{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss


    def training_step(self, batch, batch_idx):
        # Get loss for spectra-to-structure stage
        batch_size = len(batch['molecules'])
        outputs = self.model(batch)
        
        loss, loss_info = self.model.compute_loss(outputs, batch)

        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        for name, value in loss_info.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def get_optimizer_params(self):
        """
        Get optimizers based on the training stage
        """
        return {
            'lr': 1.0e-5,
        }

    def configure_optimizers(self):
        # Create optimizer
        opt_params = self.get_optimizer_params()
        optimizer = torch.optim.Adam(
            self.parameters(),
            **opt_params
        )            
        return {
                'optimizer': optimizer
            }