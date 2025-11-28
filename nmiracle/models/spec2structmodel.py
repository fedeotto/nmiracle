import torch
import torch.nn as nn
from nmiracle.models.components.multispectra_encoder import MultiSpectralEncoder, MeanPool
from common.eval_utils import (canonicalize_smiles, 
                                calculate_tanimoto_similarity, 
                                is_valid_smiles, 
                                compute_levenshtein_distance, 
                                compute_mces_distance, 
                                calculate_maccs_similarity, 
                                calculate_rdkit_similarity)
import torch.nn.functional as F
from nmiracle.models.components.transformer_models import TransformerModel
from nmiracle.models.base_model import BaseModel    

class Spectra2Structure(BaseModel):
    """Spectra to Structure model"""
    def __init__(
        self,
        tokenizer,
        structure_model_kwargs,
        multispectra_encoder_kwargs,
        pretrained_structure_path=None

    ):
        super().__init__(
            tokenizer=tokenizer,
            structure_model_kwargs=structure_model_kwargs,
            pretrained_structure_path=pretrained_structure_path
        )

        self.spectrum_encoder = MultiSpectralEncoder(**multispectra_encoder_kwargs)
        spectrum_dim = self.spectrum_encoder.d_model
        structure_dim = self.structure_model.d_model

        if spectrum_dim != structure_dim:
            self.spectrum_to_structure_adapter = nn.Linear(spectrum_dim, structure_dim)
        else:
            self.spectrum_to_structure_adapter = nn.Identity()

    def encode(self, batch):
        """Encode spectral data into structure features"""
        spectra = batch['spectra']
        spectrum_features, spectrum_mask = self.spectrum_encoder(spectra)
        structure_features = self.spectrum_to_structure_adapter(spectrum_features)
        return structure_features, spectrum_mask

    
    def forward(self, batch):
        """Forward pass"""
        structure_features, structure_features_mask = self.encode(batch)
        decoder_input = batch['decoder_input']

        structure_logits = self.structure_model(
            structure_features,
            decoder_input,
            src_key_padding_mask=structure_features_mask
        )

        return {
            'structure_features': structure_features,
            'structure_features_mask': structure_features_mask,
            'structure_logits': structure_logits
        }
    
    def compute_loss(self, outputs, batch):
        """Reuse base class method"""
        structure_loss = self.compute_structure_loss(outputs, batch)
        return structure_loss, {'structure_loss': structure_loss.item()}


class MultiTaskSpectra2Structure(Spectra2Structure):
    
    def __init__(
        self,
        tokenizer,
        structure_model_kwargs,
        multispectra_encoder_kwargs,
        num_substructures=991,
        max_substructures_count=232,
        pretrained_structure_path=None
    ):

        super().__init__(
            tokenizer=tokenizer,
            structure_model_kwargs=structure_model_kwargs,
            multispectra_encoder_kwargs=multispectra_encoder_kwargs,
            pretrained_structure_path=pretrained_structure_path
        )

        self.num_substructures = num_substructures
        self.max_substructures_count = max_substructures_count

        self.substructure_loss_fn = nn.CrossEntropyLoss()
        self.pooler = MeanPool(self.spectrum_encoder.d_model)

        self.count_classifier = nn.Sequential(
            nn.Linear(self.spectrum_encoder.d_model + num_substructures, 256),
            nn.GELU(),
            nn.Linear(256, max_substructures_count + 1)
        )
    
    def forward(self, batch):
        """Extending base forward with substructure prediction"""
        outputs = super().forward(batch)

        #Add substructure prediction
        pooled = self.pooler(outputs['structure_features'], padding_mask=outputs['structure_features_mask'])
        batch_size = pooled.size(0)
        identity_matrix = torch.eye(self.num_substructures, device=pooled.device)
        one_hot = identity_matrix.unsqueeze(0).expand(batch_size, -1, -1)

        expanded_features = pooled.unsqueeze(1).expand(-1, self.num_substructures, -1)
        combined = torch.cat([
            expanded_features.reshape(batch_size * self.num_substructures, -1),
            one_hot.reshape(batch_size * self.num_substructures, -1)
        ], dim=1)

        count_logits = self.count_classifier(combined)
        count_logits = count_logits.view(batch_size, self.num_substructures, self.max_substructures_count + 1)
        
        outputs['substructure_count_logits'] = count_logits

        return outputs

    
    def compute_loss(self, outputs, batch):
        total_loss, loss_info = super().compute_loss(outputs, batch)

        # Add substructure loss
        count_logits = outputs['substructure_count_logits']
        count_targets = torch.clamp(batch['substructures'], min=0, max=self.max_substructures_count).long()

        batch_size = count_logits.shape[0]
        flattened_logits = count_logits.view(batch_size * self.num_substructures, -1)
        flattened_targets = count_targets.view(batch_size * self.num_substructures)
        
        substructure_loss = self.substructure_loss_fn(flattened_logits, flattened_targets)

        total_loss += substructure_loss
        loss_info['substructure_loss'] = substructure_loss.item()

        return total_loss, loss_info

    
    def calculate_substructure_metrics(self, outputs, batch):
        """Substructure-specific metrics"""
        count_logits = outputs['substructure_count_logits']
        count_targets = torch.clamp(batch['substructures'], min=0, max=self.max_substructures_count).long()
        count_preds = torch.argmax(count_logits, dim=-1)
        
        correct_counts = (count_preds == count_targets).float()
        count_accuracy = correct_counts.mean().item()
        
        presence_targets = (count_targets > 0).float()
        presence_preds = (count_preds > 0).float()
        
        tp = torch.sum(presence_preds * presence_targets).item()
        fp = torch.sum(presence_preds * (1 - presence_targets)).item()
        fn = torch.sum((1 - presence_preds) * presence_targets).item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'count_accuracy': count_accuracy,
            'presence_precision': precision,
            'presence_recall': recall,
            'presence_f1': f1
        }


# class BaseSpectra2Structure(nn.Module):
#     """Base class for spectra-to-structure models"""
#     def __init__(
#         self,
#         tokenizer=None,
#         pretrained_structure_path=None,
#         structure_model_kwargs=None,
#     ):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.vocab_size = tokenizer.vocab_size

#         if pretrained_structure_path is not None:
#             print(f"Loading pretrained structure model from {pretrained_structure_path}")
#             checkpoint = torch.load(pretrained_structure_path, map_location='cpu', weights_only=False)
            
#             structure_model_kwargs = checkpoint['hyper_parameters'].model.structure_model
#             self.structure_model = TransformerModel(
#                 vocab_size=self.vocab_size,
#                 pad_token=tokenizer.pad_token_id,
#                 **structure_model_kwargs
#             )
            
#             # Load pretrained weights by removing the prefix
#             if 'state_dict' in checkpoint:
#                 # Filter and rename keys to match structure model
#                 structure_model_dict = {}
#                 for key, value in checkpoint['state_dict'].items():
#                     if key.startswith('model.structure_model.'):
#                         new_key = key.replace('model.structure_model.', '')
#                         structure_model_dict[new_key] = value
            
#                 # Load the state dict
#                 missing, unexpected = self.structure_model.load_state_dict(structure_model_dict, strict=True)
                
#                 if missing:
#                     print(f"Warning: {len(missing)} keys are missing from checkpoint:")
#                     print(missing[:5], "..." if len(missing) > 5 else "")
                
#                 if unexpected:
#                     print(f"Warning: {len(unexpected)} unexpected keys in checkpoint:")
#                     print(unexpected[:5], "..." if len(unexpected) > 5 else "")
                
#                 print(f"Loaded structure model from structure checkpoint ✅")
#         else:
#             # Structure prediction model - always needed
#             self.structure_model = TransformerModel(
#                 vocab_size=self.vocab_size,
#                 pad_token=tokenizer.pad_token_id,
#                 **structure_model_kwargs
#                 )

#             self.structure_model.initialize_weights()  # Initialize weights for the structure model

#     def forward(self, batch):
#         pass

#     def compute_loss(self, outputs, batch):
#         pass

#     @torch.no_grad()
#     def generate(self, structure_features=None, structure_features_mask=None, max_length=75, temperature=1.0, top_k=5, num_sequences=1):
#         """Generate multiple sequences per input using top-k sampling with temperature scaling"""
#         encoder_features = structure_features
#         batch_size = encoder_features.size(0)
#         device = encoder_features.device
        
#         # Create expanded batch for parallel generation
#         expanded_encoder_features = encoder_features.repeat_interleave(num_sequences, dim=0)
#         expanded_batch_size = batch_size * num_sequences
        
#         encoder_padding_mask = structure_features_mask
#         expanded_encoder_padding_mask = encoder_padding_mask.repeat_interleave(num_sequences, dim=0)
        
#         # Initialize output with start token
#         decoder_input = torch.full(
#             (expanded_batch_size, 1), 
#             self.tokenizer.start_token_id, 
#             dtype=torch.long, 
#             device=device
#         )
        
#         # Track sequence probabilities and lengths
#         sequence_neg_log_probs = torch.zeros(expanded_batch_size, dtype=torch.float, device=device)
#         sequence_lengths = torch.zeros(expanded_batch_size, dtype=torch.long, device=device)

#         # Track which sequences are finished
#         finished_sequences = torch.zeros(expanded_batch_size, dtype=torch.bool, device=device)

#         # Generate tokens one by one
#         for _ in range(max_length - 1):
#             # Forward pass with current decoder input
#             logits = self.structure_model(
#                 expanded_encoder_features,
#                 decoder_input,
#                 src_key_padding_mask=expanded_encoder_padding_mask
#             )
            
#             # Get next token logits (last position)
#             next_token_logits = logits[:, -1, :]

#             # Apply temperature first
#             next_token_logits = next_token_logits / temperature

#             # Apply softmax to ALL logits
#             full_probs = F.softmax(next_token_logits, dim=-1)

#             # Then do top-k on probabilities
#             top_k_probs, top_k_indices = torch.topk(full_probs, top_k, dim=-1)

#             # Renormalize
#             top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
#             # Sample from the top-k distribution
#             next_token_idx = torch.multinomial(top_k_probs, 1) #sample one token
            
#             # Get probability of selected token (gather along last dimension)
#             selected_probs = torch.gather(top_k_probs, -1, next_token_idx)
            
#             # Get actual token ID
#             next_token = torch.gather(top_k_indices, -1, next_token_idx)

#             # For finished sequences, force padding tokens
#             next_token[finished_sequences] = self.tokenizer.pad_token_id
#             selected_probs[finished_sequences] = 1.0  # Don't penalize padding tokens

#             # Update sequence probabilities (for non-finished sequences)
#             not_finished = ~finished_sequences

#             if not_finished.any():
#                 log_probs = torch.log(selected_probs[not_finished].view(-1) + 1e-8)
#                 log_probs = torch.clamp(log_probs, min=-100.0)
#                 sequence_neg_log_probs[not_finished] -= log_probs
            
#             # Update sequence lengths
#             sequence_lengths[not_finished & (next_token.squeeze(-1) != self.tokenizer.eos_token_id)] += 1
            
#             # Append token to sequences
#             decoder_input = torch.cat((decoder_input, next_token), dim=1)

#             # Mark sequences as finished if they generated EOS
#             eos_mask = (next_token.squeeze(-1) == self.tokenizer.eos_token_id) & ~finished_sequences
#             finished_sequences = finished_sequences | eos_mask

#             # Check if all sequences are finished
#             if finished_sequences.all():
#                 break
        
#         # Normalize scores by length
#         sequence_lengths = torch.clamp(sequence_lengths, min=1)
#         normalized_scores = sequence_neg_log_probs / sequence_lengths.float()
        
#         # Pad sequences to max_length
#         if decoder_input.size(1) < max_length:
#             padding = torch.full(
#                 (expanded_batch_size, max_length - decoder_input.size(1)),
#                 self.tokenizer.pad_token_id,
#                 dtype=torch.long,
#                 device=device
#             )
#             decoder_input = torch.cat((decoder_input, padding), dim=1)
#         else:
#             decoder_input = decoder_input[:, :max_length]
        
#         # Reshape output
#         all_sequences = decoder_input.view(batch_size, num_sequences, max_length)
#         all_scores = normalized_scores.view(batch_size, num_sequences)
        
#         return all_sequences, all_scores

#     @torch.no_grad()    
#     def evaluate_generation(self, 
#                             structure_features, 
#                             structure_features_mask, 
#                             true_smiles, 
#                             temperature=1.0,
#                             top_k=5,
#                             num_sequences=15,
#                             max_samples=None,
#                             max_molecule_len=45, 
#                             log_prefix='val'):

#         """Evaluate generation performance during training."""                    
#         batch_size = len(true_smiles)
        
#         if max_samples is None:
#             sample_size = batch_size
#         else:
#             sample_size = min(max_samples, batch_size)
        
#         sample_indices = list(range(sample_size))
        
#         # Sample the structure_features and true_smiles to match the sample size
#         sampled_structure_features = structure_features[sample_indices] 
#         sampled_structure_features_mask = structure_features_mask[sample_indices]

#         sample_true_smiles = [true_smiles[i] for i in sample_indices]

#         # Initialize metrics storage
#         metrics = {
#             'valid_rate': 0.0,
#             'tanimoto_similarity': 0.0,
#             'rdkit_similarity': 0.0,
#             'maccs_similarity': 0.0,
#             'exact_match_rate': 0.0,
#             'levenshtein_distance': 0.0,
#             'mces_distance': 0.0,
#             'top_1_accuracy': 0.0,
#             'top_5_accuracy': 0.0,
#             'top_10_accuracy': 0.0,
#             'top_15_accuracy': 0.0
#         }
        
#         try:
#             # Generate molecules using the model's generate method with precomputed features
#             generated_ids, neg_log_probs = self.generate(
#                 structure_features=sampled_structure_features,
#                 structure_features_mask= sampled_structure_features_mask,
#                 max_length=max_molecule_len,  # Adjust as needed
#                 temperature=temperature,
#                 top_k=top_k,
#                 num_sequences=num_sequences
#             )
            
#             # Reshape generated_ids into [batch_size, num_sequences, seq_len]
#             batch_size, num_seqs, seq_len = generated_ids.shape
            
#             # Decode all generated sequences for each sample
#             generated_smiles_lists = []
#             generated_scores_lists = []  # Store the scores (negative log probs)

#             for i in range(batch_size):
#                 sample_ids = generated_ids[i]
#                 sample_scores = neg_log_probs[i]
#                 decoded_smiles = []
                
#                 for j, seq_ids in enumerate(sample_ids):
#                     # Extract tokens until EOS or end of sequence
#                     tokens = []
#                     for token in seq_ids:
#                         if token.item() == self.tokenizer.eos_token_id:
#                             break
#                         if token.item() in [self.tokenizer.pad_token_id, self.tokenizer.start_token_id]:
#                             continue
#                         tokens.append(token.item())
#                     # Decode to SMILES
#                     try:
#                         smiles = self.tokenizer.decode(tokens)
#                         decoded_smiles.append((smiles, sample_scores[j].item()))  # Store (SMILES, score) tuples
#                     except Exception:
#                         decoded_smiles.append(("", float('inf')))  # Use infinity for invalid SMILES
                
#                 # Sort by score (lower is better for negative log probs)
#                 sorted_smiles = sorted(decoded_smiles, key=lambda x: x[1])
                
#                 # Store only the SMILES strings, keeping the sorted order
#                 generated_smiles_lists.append([s[0] for s in sorted_smiles])
#                 # Store the scores separately
#                 generated_scores_lists.append([s[1] for s in sorted_smiles])
            
#             # Calculate various metrics
#             valid_count = 0
#             tanimoto_scores = []
#             rdkit_scores = []
#             maccs_scores = []
#             exact_match_count = 0
#             levenshtein_distances = []
#             mces_scores = []
#             top_1_hits = 0
#             top_5_hits = 0
#             top_10_hits = 0
#             top_15_hits = 0
            
#             # If testing, store all the generated vs true molecules
#             # if log_prefix == "test":
#             #     all_canon_generated = []
#             #     all_canon_true = []
            
#             for i, (gen_smiles_list, true_smiles) in enumerate(zip(generated_smiles_lists, sample_true_smiles)):
#                 # Canonicalize all generated SMILES and filter invalid ones
#                 canon_gen_smiles = []
                
#                 for gen_smiles in gen_smiles_list:
#                     canon_gen = canonicalize_smiles(gen_smiles)
#                     if canon_gen and is_valid_smiles(canon_gen):
#                         canon_gen_smiles.append(canon_gen)
                
#                 # Get canonical version of true SMILES
#                 canon_true = canonicalize_smiles(true_smiles)
                
#                 # If testing, store all generated SMILES
#                 if log_prefix == "test":
#                     all_canon_true.append(canon_true)
#                     all_canon_generated.append(canon_gen_smiles)
                
#                 # Check if we have at least one valid molecule
#                 if canon_gen_smiles:
#                     valid_count += 1
                    
#                     # Calculate similarities and distances for each valid generated molecule
#                     best_tanimoto = 0
#                     best_rdkit = 0
#                     best_maccs = 0
#                     best_levenshtein = float('inf')
#                     best_mces = float('inf')
#                     exact_match = False
                    
#                     for j, canon_gen in enumerate(canon_gen_smiles):
#                         # Check for exact match
#                         if canon_gen == canon_true:
#                             exact_match = True
                            
#                         # Calculate Tanimoto similarity (Morgan fingerprints)
#                         tanimoto = calculate_tanimoto_similarity(canon_gen, canon_true)
#                         if tanimoto > best_tanimoto:
#                             best_tanimoto = tanimoto
                        
#                         # Calculate RDKit fingerprint similarity
#                         rdkit_sim = calculate_rdkit_similarity(canon_gen, canon_true)
#                         if rdkit_sim > best_rdkit:
#                             best_rdkit = rdkit_sim
                        
#                         # Calculate MACCS fingerprint similarity
#                         maccs_sim = calculate_maccs_similarity(canon_gen, canon_true)
#                         if maccs_sim > best_maccs:
#                             best_maccs = maccs_sim
                        
#                         # Calculate Levenshtein distance
#                         levenshtein = compute_levenshtein_distance(canon_gen, canon_true)
#                         if levenshtein < best_levenshtein:
#                             best_levenshtein = levenshtein
                        
#                         # Calculate MCES similarity
#                         mces = compute_mces_distance(canon_gen, canon_true)
#                         if mces < best_mces:
#                             best_mces = mces

#                     # Add best scores to overall metrics
#                     tanimoto_scores.append(best_tanimoto)
#                     rdkit_scores.append(best_rdkit)
#                     maccs_scores.append(best_maccs)
#                     levenshtein_distances.append(best_levenshtein)
#                     mces_distances.append(best_mces)

#                     if exact_match:
#                         exact_match_count += 1
                    
#                     # Calculate top-k accuracy
#                     if canon_true in canon_gen_smiles[:1]:
#                         top_1_hits += 1
#                     if canon_true in canon_gen_smiles[:5]:
#                         top_5_hits += 1
#                     if canon_true in canon_gen_smiles[:10]:
#                         top_10_hits += 1
#                     if canon_true in canon_gen_smiles[:15]:
#                         top_15_hits += 1
            
#             # Calculate overall metrics
#             metrics['valid_rate'] = valid_count / sample_size
#             metrics['exact_match_rate'] = exact_match_count / sample_size
#             metrics['top_1_accuracy'] = top_1_hits / sample_size
#             metrics['top_5_accuracy'] = top_5_hits / sample_size
#             metrics['top_10_accuracy'] = top_10_hits / sample_size
#             metrics['top_15_accuracy'] = top_15_hits / sample_size
            
#             # Average metrics
#             if tanimoto_scores:
#                 metrics['tanimoto_similarity'] = sum(tanimoto_scores) / len(tanimoto_scores)
#             if rdkit_scores:
#                 metrics['rdkit_similarity'] = sum(rdkit_scores) / len(rdkit_scores)
#             if maccs_scores:
#                 metrics['maccs_similarity'] = sum(maccs_scores) / len(maccs_scores)
#             if levenshtein_distances:
#                 metrics['levenshtein_distance'] = sum(levenshtein_distances) / len(levenshtein_distances)
#             if mces_scores:
#                 metrics['mces_similarity'] = sum(mces_scores) / len(mces_scores)
                
#             return metrics
        
#         except Exception as e:
#             print(f"Error during generation evaluation: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             raise e


# class Spectra2Structure(BaseSpectra2Structure):
#     """Spectra to Structure model"""
#     def __init__(self, 
#                  tokenizer=None,
#                  pretrained_spectrum_encoder_path=None,
#                  pretrained_structure_path=None,
#                  freeze_spectrum_encoder=False,
#                  multispectra_encoder_kwargs=None,
#                  structure_model_kwargs=None):
        
#         super().__init__(
#             tokenizer=tokenizer,
#             pretrained_structure_path=pretrained_structure_path,
#             structure_model_kwargs=structure_model_kwargs
#         )

#         self.structure_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

#         # Load and initialize spectrum encoder from alignment checkpoint
#         if pretrained_spectrum_encoder_path is not None:
#             print(f"Loading pretrained spectrum encoder model from {pretrained_spectrum_encoder_path}")
#             checkpoint = torch.load(pretrained_spectrum_encoder_path, map_location='cpu', weights_only=False)
            
#             encoder_kwargs = checkpoint['hyper_parameters'].model.multispectra_encoder
#             self.spectrum_encoder = MultiSpectralEncoder(**encoder_kwargs)

#             # Load pretrained weights
#             spectrum_encoder_dict = {}
#             for key, value in checkpoint['state_dict'].items():
#                 if key.startswith('model.spectrum_encoder.'):
#                     new_key = key.replace('model.spectrum_encoder.', '')
#                     spectrum_encoder_dict[new_key] = value

#             # Load the state dict
#             missing_spec, unexpected_spec = self.spectrum_encoder.load_state_dict(spectrum_encoder_dict, strict=True)
#             if missing_spec:
#                 print(f"Warning: {len(missing_spec)} keys are missing from checkpoint:")
#                 print(missing_spec[:5], "..." if len(missing_spec) > 5 else "")
            
#             if unexpected_spec:
#                 print(f"Warning: {len(unexpected_spec)} unexpected keys in checkpoint:")
#                 print(unexpected_spec[:5], "..." if len(unexpected_spec) > 5 else "")
#             else:
#                 print(f"Loaded spectrum encoder from checkpoint ✅")
#         else:
#             print("No pretrained alignment model provided, initializing spectrum encoder with default provided parameters.")
#             self.spectrum_encoder = MultiSpectralEncoder(**multispectra_encoder_kwargs)

#         # Freeze spectrum encoder if specifiedx
#         if freeze_spectrum_encoder:
#             for param in self.spectrum_encoder.parameters():
#                 param.requires_grad = False

#         # Create adapter if dimensions don't match
#         spectrum_dim = self.spectrum_encoder.d_model
#         structure_dim = self.structure_model.d_model
        
#         if spectrum_dim != structure_dim:
#             print(f"Creating adapter from dimension {spectrum_dim} to {structure_dim}")
#             self.spectrum_to_structure_adapter = nn.Linear(spectrum_dim, structure_dim)
#         else:
#             print(f"No adapter needed, dimensions match: {spectrum_dim}")
#             self.spectrum_to_structure_adapter = nn.Identity()

#     def forward(self, batch):
#         outputs = {}
        
#         # Process spectral data
#         spectra = batch['spectra']
#         spectrum_features, spectrum_mask = self.spectrum_encoder(spectra)

#         # Adapt spectrum features to structure space if needed
#         structure_features = self.spectrum_to_structure_adapter(spectrum_features)
#         structure_features_mask = spectrum_mask

#         # Store features for generation later
#         outputs['structure_features'] = structure_features
#         outputs['structure_features_mask'] = structure_features_mask
        
#         # Get decoder inputs for teacher forcing
#         decoder_input = batch['decoder_input']
        
#         # Forward with teacher forcing using pre-created decoder input
#         structure_logits = self.structure_model(
#             structure_features,
#             decoder_input,
#             src_key_padding_mask=structure_features_mask
#         )
#         outputs['structure_logits'] = structure_logits
        
#         return outputs


#     def compute_loss(self, outputs, batch):
#         total_loss = 0.0
#         loss_info = {}

#         target = batch['target']
#         target = target.view(-1)

#         structure_logits = outputs['structure_logits']
#         structure_logits = structure_logits.view(-1, structure_logits.size(-1))

#         structure_loss = self.structure_loss_fn(structure_logits, target)
#         total_loss += structure_loss
#         loss_info['structure_loss'] = structure_loss.item()

#         return total_loss, loss_info
    

# class MultiTaskSpectra2Structure(BaseSpectra2Structure):
#     """Spectra to Structure model with multi-task learning"""
#     def __init__(self, 
#                  tokenizer=None,
#                  pretrained_spectrum_encoder_path=None,
#                  pretrained_structure_path=None,
#                  freeze_spectrum_encoder=False,
#                  multispectra_encoder_kwargs=None,
#                  num_substructures=991,
#                  max_substructures_count=232,
#                  structure_model_kwargs=None):
        
#         super().__init__(
#             tokenizer=tokenizer,
#             pretrained_structure_path=pretrained_structure_path,
#             structure_model_kwargs=structure_model_kwargs
#         )

#         self.structure_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#         self.substructure_loss_fn = nn.CrossEntropyLoss()
#         self.num_substructures = num_substructures
#         self.max_substructures_count = max_substructures_count

#         # Initialize the multi-spectra encoder
#         self.spectrum_encoder = MultiSpectralEncoder(**multispectra_encoder_kwargs)
#         self.count_classifier = nn.Sequential(
#             nn.Linear(self.spectrum_encoder.d_model + num_substructures, 256),
#             nn.GELU(),
#             nn.Linear(256, max_substructures_count + 1)
#             )
            
#         #Pooler defined anyway (not learnable)
#         self.pooler = MeanPool(multispectra_encoder_kwargs['d_model'])

#         # Freeze spectrum encoder if specified
#         if freeze_spectrum_encoder:
#             for param in self.spectrum_encoder.parameters():
#                 param.requires_grad = False

#         # Create adapter if dimensions don't match
#         spectrum_dim = self.spectrum_encoder.d_model
#         structure_dim = self.structure_model.d_model
        
#         if spectrum_dim != structure_dim:
#             print(f"Creating adapter from dimension {spectrum_dim} to {structure_dim}")
#             self.spectrum_to_structure_adapter = nn.Linear(spectrum_dim, structure_dim)
#         else:
#             print(f"No adapter needed, dimensions match: {spectrum_dim}")
#             self.spectrum_to_structure_adapter = nn.Identity()

#     def forward(self, batch):
#         outputs = {}
        
#         # Process spectral data
#         spectra = batch['spectra']
#         spectrum_features, spectrum_mask = self.spectrum_encoder(spectra)

#         # Adapt spectrum features to structure space if needed
#         structure_features = self.spectrum_to_structure_adapter(spectrum_features)
#         structure_features_mask = spectrum_mask

#         # Store features for generation later
#         outputs['structure_features'] = structure_features
#         outputs['structure_features_mask'] = structure_features_mask

#         decoder_input = batch['decoder_input']

#         # Forward with teacher forcing using pre-created decoder input
#         structure_logits = self.structure_model(
#             structure_features,
#             decoder_input,
#             src_key_padding_mask=structure_features_mask
#         )
#         outputs['structure_logits'] = structure_logits
        
#         #Substructure prediction
#         spectra_pooled = self.pooler(spectrum_features, padding_mask=spectrum_mask)
#         batch_size = spectra_pooled.size(0)
#         # Create one-hot vectors for all substructures
#         identity_matrix = torch.eye(self.num_substructures, device=spectra_pooled.device)
#         one_hot = identity_matrix.unsqueeze(0).expand(batch_size, -1, -1)

#         # Expand spectral features
#         expanded_features = spectra_pooled.unsqueeze(1).expand(-1, self.num_substructures, -1)

#         # Concatenate features with one-hot vectors
#         combined = torch.cat([
#             expanded_features.reshape(batch_size * self.num_substructures, -1),
#             one_hot.reshape(batch_size * self.num_substructures, -1)
#         ], dim=1)

#         # Predict count distributions for all substructures
#         count_logits = self.count_classifier(combined)

#         # Reshape to [batch_size, num_substructures, max_count_value]
#         count_logits = count_logits.view(batch_size, self.num_substructures, self.max_substructures_count +1)
#         outputs['substructure_count_logits'] = count_logits

#         return outputs
    
#     def compute_loss(self, outputs, batch):
#         total_loss = 0.0
#         loss_info = {}

#         target = batch['target']
#         target = target.view(-1)

#         structure_logits = outputs['structure_logits']
#         structure_logits = structure_logits.view(-1, structure_logits.size(-1))

#         structure_loss = self.structure_loss_fn(structure_logits, target)
#         total_loss += structure_loss
#         loss_info['structure_loss'] = structure_loss.item()

#         # Substructure count prediction loss with class weighting
#         count_logits = outputs['substructure_count_logits']
#         count_targets = torch.clamp(batch['substructures'], min=0, max=self.max_substructures_count).long()

#         batch_size = count_logits.shape[0]
#         flattened_logits = count_logits.view(batch_size * self.num_substructures, -1)
#         flattened_targets = count_targets.view(batch_size * self.num_substructures)

#         substructure_loss = self.substructure_loss_fn(flattened_logits, flattened_targets)

#         total_loss += substructure_loss
#         loss_info['substructure_loss'] = substructure_loss.item()

#         return total_loss, loss_info

#     def calculate_substructure_metrics(self, outputs, batch):
#         metrics = {}

#         #get count predictions
#         count_logits = outputs['substructure_count_logits']
#         count_targets = torch.clamp(batch['substructures'], min=0, max=self.max_substructures_count).long()

#         #Get predicted counts
#         count_preds = torch.argmax(count_logits, dim=-1)  # Shape: [batch_size, num_substructures]

#         #Calculate count accuracy
#         correct_counts = (count_preds == count_targets).float()
#         count_accuracy = correct_counts.mean().item()

#         #Calculate binary presence accuracy (count >0)
#         presence_targets = (count_targets>0).float()
#         presence_preds = (count_preds>0).float()

#         # True positives, flase positives, false negatives
#         tp = torch.sum(presence_preds * presence_targets).item()
#         fp = torch.sum(presence_preds * (1 - presence_targets)).item()
#         fn = torch.sum((1 - presence_preds) * presence_targets).item()

#         # Calculate precision, recall, and F1
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

#         # Return all metrics
#         metrics = {
#             'count_accuracy': count_accuracy,
#             'presence_precision': precision,
#             'presence_recall': recall,
#             'presence_f1': f1
#         }

#         return metrics