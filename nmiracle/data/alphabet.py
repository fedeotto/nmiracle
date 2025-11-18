import os
import numpy as np
import lmdb
import pickle
import zlib
from nmiracle.data.tokenizer import BasicSmilesTokenizer
from typing import Set, Optional
from omegaconf import DictConfig
from pathlib import Path


def generate_alphabet(
    config: DictConfig,
    output_path: Optional[str] = None,
    tokenizer: Optional[BasicSmilesTokenizer] = None,
) -> np.ndarray:
    """Generate alphabet from the appropriate dataset based on config"""
    
    if tokenizer is None:
        tokenizer = BasicSmilesTokenizer()
    
    print("Generating comprehensive SMILES alphabet...")
    
    unique_tokens = set()
    smiles_array_path = os.path.join(config.data_dir, 'smiles.npy')
    smiles_array = np.load(smiles_array_path, allow_pickle=True)
    smiles_array = [smi.decode('utf-8') if isinstance(smi, bytes) else smi for smi in smiles_array]

    for smiles in smiles_array:
        tokens = tokenizer.tokenize(smiles)
        unique_tokens.update(tokens)
        
    print(f"Added {len(unique_tokens)} unique tokens")
    
    # Create sorted alphabet array
    alphabet = np.array(sorted(list(unique_tokens)))
    print(f"Generated alphabet with {len(alphabet)} tokens")
    
    # Save alphabet if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, alphabet)
        print(f"Saved alphabet to {output_path}")
    
    return alphabet

def generate_or_load_alphabet(
    config: DictConfig,
    alphabet_path: Optional[str] = None,
    tokenizer: Optional[BasicSmilesTokenizer] = None,
) -> np.ndarray:
  
    # Check if we should use existing alphabet
    if alphabet_path and os.path.exists(alphabet_path):
        print(f"Loading existing alphabet from {alphabet_path}")
        return np.load(alphabet_path)

    cfg = config.data if hasattr(config, 'data') else config
    return generate_alphabet(cfg, alphabet_path, tokenizer)