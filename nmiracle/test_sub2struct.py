import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import OmegaConf
from common.eval_utils import canonicalize_smiles, is_valid_smiles
from nmiracle.models.pl_module import Sub2StructModule
from nmiracle.data.datamodule import SpectralDataModule
from nmiracle.data.tokenizer import BasicSmilesTokenizer
import os
from tqdm import tqdm
import numpy as np
import argparse

os.environ['HYDRA_FULL_ERROR']     = '1'

# ============================================================================
# Default Configuration
# ============================================================================
MODEL_PATH = 'nmiracle/ckpts/sub2struct'
CKPT_NAME  = "epoch=439-val_loss=0.04.ckpt"
NUM_WORKERS = 0  # Set to 0 for no multiprocessing, adjust as needed
PREFETCH_FACTOR = None  # Set to None for no prefetching, adjust as needed
BATCH_SIZE = 8  # Adjust batch size as needed
TEMPERATURE = 1.0
TOP_K = 5
NUM_SEQUENCES = 3
MAX_MOLECULES = None  # None means use all test samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--checkpoint", type=str, default=CKPT_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--num_sequences", type=int, default=NUM_SEQUENCES)
    parser.add_argument("--max_molecules", type=int, default=MAX_MOLECULES)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()

def load_model_and_checkpoint(model_path, ckpt_name):
    """Load both config and model state from checkpoint"""
    model_path = Path(model_path)
    
    # Find checkpoint path
    ckpt_path = model_path / "checkpoints" / ckpt_name
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
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model configuration and checkpoint
    config, checkpoint = load_model_and_checkpoint(args.model_path, args.checkpoint)

    OmegaConf.set_struct(config, False)

    config.paths.hydra_dir = str(Path(args.model_path) / ".hydra")
    config.paths.output_dir = str(Path(args.model_path))
    config.paths.work_dir = str(Path(args.model_path))

    config.data.num_workers = {'train': 0, 'val': 0, 'test': args.num_workers}
    config.data.prefetch_factor = {'train': None, 'val': None, 'test': PREFETCH_FACTOR}
    config.data.batch_size.test = args.batch_size

    config.data.max_substructures_count = config.data.max_count_value 
    config.model.pretrained_structure_model_path = None
    config.paths.data_dir = 'datasets'          #a few adjustment (things called with different names when model was trained.)
    config.data.data_dir = str(Path(config.paths.data_dir) / "pretrain")

    checkpoint['state_dict']['model.substructure_count_embedding.weight']  \
        = checkpoint['state_dict']['model.count_embedding.weight'].clone()

    del checkpoint['state_dict']['model.count_embedding.weight']

    #loading alphabet and metadata from model path
    alphabet = np.load(Path(args.model_path) / "alphabet.npy", allow_pickle=True)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Setup tokenizer
    tokenizer = BasicSmilesTokenizer()
    tokenizer.setup_alphabet(alphabet)

    # Create model instance (without loading checkpoint)
    pl_module = Sub2StructModule(
        config=config,
        tokenizer=tokenizer
    )

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

    max_molecules = args.max_molecules  # None means use all test samples
    
    pl_module.eval()

    mol_count = 0

    generations = {
        'all_canon_true': [],
        'all_canon_generated': [],
    }

    for i, batch in enumerate(tqdm(test_dataloader)):
        batch_size = len(batch['molecules'])
        mol_count += batch_size

        batch['substructures'] = batch['substructures'].to(device)
        batch['substructure_counts'] = batch['substructure_counts'].to(device)
        batch['decoder_input'] = batch['decoder_input'].to(device)
        batch['target'] = batch['target'].to(device)

        if max_molecules is not None and mol_count >= max_molecules:
            break
        
        with torch.enable_grad():
            outputs = pl_module.model(batch)

        structure_features = outputs['structure_features']
        structure_features_mask = outputs['structure_features_mask']
        true_smiles = batch['molecules']

        generated_ids, neg_log_probs = pl_module.model.generate(
            structure_features=structure_features,
            structure_features_mask=structure_features_mask,
            max_length = data_module.test_dataset.max_molecule_len,
            temperature=args.temperature,
            top_k=args.top_k,
            num_sequences=args.num_sequences
        )


        batch_size, num_seqs, seq_len = generated_ids.shape
        
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

    import json
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path(args.model_path) / f"test_results-temperature-{args.temperature}-top_k-{args.top_k}-num_seqs-{args.num_sequences}.json"

    results = {
        'generations': generations,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'num_sequences': args.num_sequences
    }

    #write results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()