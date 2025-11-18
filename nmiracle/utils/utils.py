from rdkit import Chem
import torch
import numpy as np
import selfies as sf
import random

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def flatten_dataset_config(config):
    """
    Flatten dataset-specific config to top level for easy access
    
    This keeps your existing code working while adding flexibility
    """
    from omegaconf import OmegaConf
    
    # Get dataset type
    dataset_type = getattr(config, 'dataset_type', getattr(config, 'dataset_name', 'alberts'))
    
    # Get the dataset-specific config
    if hasattr(config, dataset_type):
        dataset_config = getattr(config, dataset_type)
        
        # Create a copy of the original config
        flattened = OmegaConf.create(config)
        
        # Merge dataset-specific settings to top level
        for key, value in dataset_config.items():
            flattened[key] = value
            
        return flattened
    else:
        # If no dataset-specific config found, return as-is
        return config
    
def get_wandb_name(config):
    # Start with base model name
    name_parts = [config.model_name, config.data.training_stage, config.data.dataset_name]
    
    # Add active modalities with underscore separator
    modality_parts = []
    if hasattr(config.data, 'use_ir') and config.data.use_ir:
        modality_parts.append('ir')
    if hasattr(config.data, 'use_hnmr') and config.data.use_hnmr:
        modality_parts.append('hnmr')
    if hasattr(config.data, 'use_cnmr') and config.data.use_cnmr:
        modality_parts.append('cnmr')
    
    # Add modalities if available (with underscore separator and trailing underscore)
    if modality_parts:
        name_parts.append('_'.join(modality_parts))
    
    # Add batch size if available
    if hasattr(config.data, 'batch_size'):
            name_parts.append(f"bs-{config.data.batch_size.train}")
    
    # Add learning rate with full precision
    lr = None
    if hasattr(config.model, 'optimizer') and hasattr(config.model.optimizer, 'lr'):
        lr = config.model.optimizer.lr
    elif hasattr(config, 'optimizer') and hasattr(config.optimizer, 'lr'):
        lr = config.optimizer.lr
    elif hasattr(config.model, 'lr_pretrain'):
        lr = config.model.lr_pretrain
        
    if lr is not None:
        # Format with full precision (e.g., 0.00005)
        name_parts.append(f"lr-{lr}")
    
    # Add formula as a separate flag
    if hasattr(config.data, 'use_formula'):
        name_parts.append(f"formula-{str(config.data.use_formula).lower()}")
    
    # Add architecture flags
    if hasattr(config.model, 'shared_encoder'):
        name_parts.append(f"shared_enc-{str(config.model.shared_encoder).lower()}")
        
    if hasattr(config.model, 'fusion_scheme'):
        name_parts.append(f"fusion-{config.model.fusion_scheme}")
    
    # Add 'continued' suffix if resuming from checkpoint
    if hasattr(config, 'resume') and config.resume.checkpoint_path:
        name_parts.append('continued')
        
    # Combine all parts with hyphens
    return '-'.join(name_parts)


            