import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from scipy.signal import find_peaks
import h5py
from pathlib import Path
import torch.nn.functional as F
from rdkit import Chem
import os

class AlbertsDataset(Dataset):
    """
    Dataset for spectra-to-structure training using the Alberts dataset
    """
    def __init__(
        self,
        data_dir,
        indices,
        tokenizer=None,
        n_ir_features=1800,
        n_hnmr_features=10000,
        n_cnmr_features=10000,
        max_molecule_len=128,
        use_ir=False,
        use_hnmr=True,
        use_cnmr=True,
        cnmr_binary=True,
        cnmr_binary_bins=80
    ):
        self.n_ir_features = n_ir_features
        self.n_hnmr_features = n_hnmr_features
        self.n_cnmr_features = n_cnmr_features
        self.use_ir = use_ir
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.cnmr_binary = cnmr_binary
        self.cnmr_binary_bins = cnmr_binary_bins
        self.tokenizer = tokenizer
        self.max_molecule_len = max_molecule_len
        self.data_dir = data_dir

        self.indices = indices

        #load SMILES
        smiles_path = os.path.join(data_dir, "smiles.npy")
        self.smiles = np.load(smiles_path, allow_pickle=True)
        self.smiles = [smi.decode('utf-8') if isinstance(smi, bytes) else smi for smi in self.smiles]

        # Store paths to files (we'll open them when needed)
        self.spectra_path = os.path.join(data_dir, "spectra.h5")
        self.spectra_file = None

        self.substructures_path = os.path.join(data_dir, "substructure_counts.h5")
        self.substructures_file = None  # h5py.File(self.substructures_path, "r")#['counts'][:] #this can be a bit memory-heavy

        print(f"Loaded {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)

    def _open_spectra_file(self):
        """Open HDF5 file lazily"""
        if self.spectra_file is None:
            self.spectra_file = h5py.File(self.spectra_path, 'r')
        return self.spectra_file

    def _open_substructures_file(self):
        if self.substructures_file is None:
            self.substructures_file = h5py.File(self.substructures_path, 'r')
        return self.substructures_file
    
    def _process_spectrum(self, spectrum):
        """Process spectra in NMR2Struct style by concatenating them"""
        # Prepare concatenated spectral input as in NMR2Struct
        spectral_components = []
        
        ir = spectrum[:1800]
        hnmr = spectrum[1800:11800]
        cnmr = spectrum[11800:]

        if self.use_ir:
            if len(ir) > self.n_ir_features:
                ir = ir[:self.n_ir_features]
            else:
                ir = np.pad(ir, (0, self.n_ir_features - len(ir)), 'constant')
            
            # Normalize
            if np.max(np.abs(ir)) > 0:
                ir = ir / np.max(np.abs(ir))
            
            spectral_components.append(ir)
        
        if self.use_hnmr:
            # Truncate or pad
            if len(hnmr) > self.n_hnmr_features:
                hnmr = hnmr[:self.n_hnmr_features]
            else:
                hnmr = np.pad(hnmr, (0, self.n_hnmr_features - len(hnmr)), 'constant')
            
            # Normalize
            if np.max(np.abs(hnmr)) > 0:
                hnmr = hnmr / np.max(np.abs(hnmr))
            
            spectral_components.append(hnmr)
        
        if self.use_cnmr:            
            if self.cnmr_binary:
                # Use the same approach as NMR2Struct for processing CNMR spectra
                # If continuous spectrum (10,000 points), find peaks and convert to ppm
                if len(cnmr) == 10000:  # Continuous spectrum                    
                    # Find peaks - consider a point a peak if it's at least 10% of max intensity
                    # and at least 5 points apart from other peaks
                    peaks, _ = find_peaks(cnmr, height=0.1*np.max(cnmr), distance=5)
                    
                    # Convert peak indices to ppm values (0-220 ppm range)
                    ppm_range = np.linspace(0, 220, 10000)
                    peak_positions = ppm_range[peaks]
                else:
                    # Already a list of peak positions
                    peak_positions = np.array(cnmr)
                
                # Create binary representation with specified number of bins
                bin_edges = np.linspace(0, 220, self.cnmr_binary_bins + 1)
                binned_cnmr = np.zeros(self.cnmr_binary_bins)
                
                if len(peak_positions) > 0:
                    # Assign peaks to bins
                    bin_indices = np.digitize(peak_positions, bin_edges) - 1
                    bin_indices = np.clip(bin_indices, 0, self.cnmr_binary_bins - 1)
                    binned_cnmr[bin_indices] = 1
                cnmr = binned_cnmr
            else:
                # Continuous representation - standard processing
                if len(cnmr) > self.n_cnmr_features:
                    cnmr = cnmr[:self.n_cnmr_features]
                else:
                    cnmr = np.pad(cnmr, (0, self.n_cnmr_features - len(cnmr)), 'constant')
                
                # Normalize
                if np.max(np.abs(cnmr)) > 0:
                    cnmr = cnmr / np.max(np.abs(cnmr))
            
            spectral_components.append(cnmr)
        
        # Concatenate all spectral components
        concatenated_spectra = np.concatenate(spectral_components)
        return torch.tensor(concatenated_spectra, dtype=torch.float32)
    
    def _process_single_sample(self, idx):
        """Process a single sample (extracted from your current __getitem__)"""
        data_idx = self.indices[idx]

        spectra_file = self._open_spectra_file()
        spectrum = spectra_file['spectra'][data_idx]

        # Process the 
        # raw spectra stacked in order IR (1,800), HNMR (10,000), CNMR (10,000) in order here
        processed_spectrum = self._process_spectrum(spectrum)
        smiles = self.smiles[data_idx]

       # Tokenize SMILES (same as your existing code)
        token_ids = self.tokenizer.encode(
            smiles,
            max_length=self.max_molecule_len,
            padding=None,
            truncation=True,
            return_tensors=None
        ).input_ids


        # Create target and decoder input
        target = token_ids.copy()
        target.append(self.tokenizer.eos_token_id)
        
        decoder_input = [self.tokenizer.start_token_id]
        decoder_input.extend(token_ids)

        # Pad sequences
        if len(target) < self.max_molecule_len:
            target.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(target)))
        if len(decoder_input) < self.max_molecule_len:
            decoder_input.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(decoder_input)))

        #Get substructures
        substructures_file = self._open_substructures_file()
        substructures = substructures_file['counts'][data_idx].astype('float32')

        result = {
            'spectra': processed_spectrum,
            'molecules': smiles,
            'substructures': torch.tensor(substructures, dtype=torch.float32),
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
        }

        return result
    
    def __getitem__(self, idx):
        return self._process_single_sample(idx)


class PreTrainDataset(Dataset):    
    def __init__(
        self, 
        data_dir,
        indices,
        tokenizer,
        max_molecule_len=None,
        max_substructures_len=None,
        max_substructures_count=None,
    ):
        self.data_dir = data_dir
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_molecule_len = max_molecule_len
        self.max_substructures_count  = max_substructures_count
        self.max_substructures_len = max_substructures_len

        # Set up file paths
        self.smiles_path = Path(data_dir) / 'smiles.npy'
        self.substructures_path = Path(data_dir) / 'substructure_counts.h5'

        # Load SMILES once (always small)
        smiles = np.load(self.smiles_path, allow_pickle=True)[:]
        self.smiles = [smi.decode('utf-8') if isinstance(smi, bytes) else smi for smi in smiles]
        
        # File handles for lazy loading
        self.substructures_file = None #h5py.File(self.substructures_path, "r")
        # self.substructures = None #self.substructures_file["counts"][:]

    def _open_substructures_file(self):
        """Lazy open substructures file"""
        if self.substructures_file is None:
            self.substructures_file = h5py.File(self.substructures_path, 'r')
        return self.substructures_file
    
    def _process_smiles_tokens(self, smiles):
        """Process SMILES into tokens (shared logic)"""
        token_ids = self.tokenizer.encode(
            smiles,
            max_length=self.max_molecule_len,
            padding=None,
            truncation=True,
            return_tensors=None
        ).input_ids
        
        # Create target and decoder input
        target = token_ids.copy()
        target.append(self.tokenizer.eos_token_id)
        
        decoder_input = [self.tokenizer.start_token_id]
        decoder_input.extend(token_ids)
        
        # Pad sequences
        if len(target) < self.max_molecule_len:
            target.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(target)))
        if len(decoder_input) < self.max_molecule_len:
            decoder_input.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(decoder_input)))
        
        return {
            'target': torch.tensor(target, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'attention_mask': (torch.tensor(target, dtype=torch.long) != self.tokenizer.pad_token_id).float()
        }
    
    def _process_single_sample(self, idx):
        """Process a single sample on-demand"""
        sample_idx = self.indices[idx]
        smiles = self.smiles[sample_idx]

        # Load substructure
        substructures = self._open_substructures_file()
        substructures = substructures['counts'][sample_idx]
        # substructures = self.substructures_file["counts"][sample_idx].astype('float32')

        # Process SMILES tokens
        token_data = self._process_smiles_tokens(smiles)

        # Process substructures
        substructures, substructure_counts = self._process_substructures_and_counts(substructures)

        # Base result
        result = {
            'molecules': smiles,
            'decoder_input': token_data['decoder_input'],
            'attention_mask': token_data['attention_mask'],
            'target': token_data['target'],
            'substructures': substructures,
            'substructure_counts': substructure_counts,
        }
            
        return result

    def _process_substructures_and_counts(self, counts):
        nonzero_mask = counts > 0 
        nonzero_indices = np.arange(len(counts))[nonzero_mask] + 1
        nonzero_counts = counts[nonzero_mask]

        #Sort by index (important for consistent processing)
        sorted_indices = np.argsort(nonzero_indices)
        sorted_substructure_indices = nonzero_indices[sorted_indices]
        sorted_counts = nonzero_counts[sorted_indices]

        # Pad with zeros to max_len
        padded_indices = np.zeros(self.max_substructures_len, dtype=np.int64)
        padded_counts = np.zeros(self.max_substructures_len, dtype=np.int64)

        # Fill in actual values
        length = min(len(sorted_substructure_indices), self.max_substructures_len)
        padded_indices[:length] = sorted_substructure_indices[:length]
        padded_counts[:length] = sorted_counts[:length]

        return (
            torch.tensor(padded_indices, dtype=torch.long),
            torch.tensor(padded_counts, dtype=torch.long)
        )

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Process data on-demand
        return self._process_single_sample(idx)
    