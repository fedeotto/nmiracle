import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import OmegaConf
from common.eval_utils import is_valid_smiles, canonicalize_smiles
import rootutils
from nmiracle.models.pl_module import Spec2StructModule
from nmiracle.data.datamodule import SpectralDataModule
from nmiracle.data.tokenizer import BasicSmilesTokenizer
import os
from tqdm import tqdm
import numpy as np

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

os.environ['HYDRA_FULL_ERROR']     = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "nmiracle/ckpts/spec2struct_ir_hnmr_cnmr"
CKPT_NAME  = "epoch=295-val_loss=0.15.ckpt"
NUM_WORKERS = 0  # Set to 0 for no multiprocessing, adjust as needed
PREFETCH_FACTOR = None  # Set to None for no prefetching, adjust as needed
BATCH_SIZE = 8  # Adjust batch size as needed
TEMPERATURE = 1.0
TOP_K = 5
NUM_SEQUENCES = 3

def load_model_and_checkpoint(model_path, ckpt_name):
    """Load both config and model state from checkpoint"""
    model_path = Path(model_path)
    
    # Find checkpoint path
    ckpt_path = model_path / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        checkpoints_dir = model_path / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.ckpt"))
            if checkpoints:
                ckpt_path = checkpoints[0]
                print(f"Using {ckpt_path}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        else:
            raise FileNotFoundError(f"Checkpoints directory not found")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract config from hyperparameters
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']
        if isinstance(config, dict):
            config = OmegaConf.create(config)
    else:
        raise KeyError("No hyperparameters found in checkpoint")
    
    return config, checkpoint

def main():    
    # Load model configuration and checkpoint
    config, checkpoint = load_model_and_checkpoint(MODEL_PATH, CKPT_NAME)

    OmegaConf.set_struct(config, False)

    config.paths.hydra_dir = str(Path(MODEL_PATH) / ".hydra")
    config.paths.output_dir = str(Path(MODEL_PATH))
    config.paths.work_dir = str(Path(MODEL_PATH))

    config.data.num_workers = {'train': 0, 'val': 0, 'test': NUM_WORKERS}
    config.data.prefetch_factor = {'train': None, 'val': None, 'test': PREFETCH_FACTOR}
    config.data.batch_size.test = BATCH_SIZE

    config.data.max_substructures_count = config.data.max_count_value 
    config.model.pretrained_structure_model_path = None
    config.paths.data_dir = 'datasets'          #a few adjustment (things called with different names when model was trained.)
    config.data.data_dir = str(Path(config.paths.data_dir) / "multispectra")

    #loading alphabet and metadata from model path
    alphabet = np.load(Path(MODEL_PATH) / "alphabet.npy", allow_pickle=True)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Setup tokenizer
    tokenizer = BasicSmilesTokenizer()
    tokenizer.setup_alphabet(alphabet)

    # Create model instance (without loading checkpoint)
    pl_module = Spec2StructModule(config=config, tokenizer=tokenizer)

    pl_module.to(device)

    if 'state_dict' in checkpoint:
        # Load model state from checkpoint
        pl_module.load_state_dict(checkpoint['state_dict'], strict=True)
        print("✅ Model state loaded from checkpoint.")
    else:
        raise KeyError("No state_dict found in checkpoint")
    
    # Initialize data module
    data_module = SpectralDataModule(config=config.data, 
                                     tokenizer=tokenizer)
    data_module.setup(stage='test')

    test_dataloader = data_module.test_dataloader()

    max_molecules = 100 # use all test for now..
    
    pl_module.eval()

    mol_count = 0

    generations = {
        'all_canon_true': [],
        'all_canon_generated': [],
    }

    for i, batch in enumerate(tqdm(test_dataloader)):
        batch_size = len(batch['molecules'])
        mol_count += batch_size

        batch['spectra'] = batch['spectra'].to(device)        
        if mol_count >= max_molecules:
            break

        # enabling gradients at inference to avoid problems with nn.Transformer
        with torch.enable_grad():
            outputs = pl_module.model(batch)
        
        structure_features = outputs['structure_features']
        structure_features_mask = outputs['structure_features_mask']
        true_smiles = batch['molecules']
        
        generated_ids, neg_log_probs = pl_module.model.generate(
            structure_features=structure_features,
            structure_features_mask=structure_features_mask,
            max_length = data_module.test_dataset.max_molecule_len,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            num_sequences=NUM_SEQUENCES
        )

        batch_size = generated_ids.shape[0]

        # Decode all generated sequences for each sample
        generated_smiles_lists = []
        generated_scores_lists = []

        for i in range(batch_size):
            sample_ids = generated_ids[i]
            sample_scores = neg_log_probs[i]
            decoded_smiles = []
            
            for j, seq_ids in enumerate(sample_ids):
                # Extract tokens until EOS or end of sequence
                tokens = []
                for token in seq_ids:
                    if token.item() == tokenizer.eos_token_id:
                        break
                    if token.item() in [tokenizer.pad_token_id, tokenizer.start_token_id]:
                        continue
                    tokens.append(token.item())
                # Decode to SMILES
                try:
                    smiles = tokenizer.decode(tokens)
                    decoded_smiles.append((smiles, sample_scores[j].item()))  # Store (SMILES, score) tuples
                except Exception:
                    decoded_smiles.append(("", float('inf')))
            
            sorted_smiles = sorted(decoded_smiles, key=lambda x: x[1])
            generated_smiles_lists.append([s[0] for s in sorted_smiles])
            generated_scores_lists.append([s[1] for s in sorted_smiles])


        for i, (gen_smiles_list, true_smiles) in enumerate(zip(generated_smiles_lists,true_smiles)):
            canon_gen_smiles = []
            for gen_smiles in gen_smiles_list:
                canon_gen = canonicalize_smiles(gen_smiles)
                if canon_gen and is_valid_smiles(canon_gen):
                    canon_gen_smiles.append(canon_gen)
            
            canon_true = canonicalize_smiles(true_smiles)

            generations['all_canon_true'].append(canon_true)
            generations['all_canon_generated'].append(canon_gen_smiles)

    # Save generations and metrics to json file
    import json
    output_file = Path(MODEL_PATH) / f"test_results-temperature-{TEMPERATURE}-top_k-{TOP_K}.json"

    results = {
        'generations': generations,
        'temperature': TEMPERATURE,
        'top_k': TOP_K
    }

    #write results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()