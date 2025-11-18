import torch
from common.eval_utils import (canonicalize_smiles, 
                                calculate_tanimoto_similarity, 
                                is_valid_smiles,
                                compute_mces_distance,
                                compute_levenshtein_distance, 
                                calculate_maccs_similarity, calculate_rdkit_similarity
                                )
import torch.nn as nn
from nmiracle.models.components.transformer_models import TransformerModel
import torch.nn.functional as F
import traceback

class BaseModel(nn.Module):
    """
    Base class for all structure-generation models.
    """
    def __init__(self, tokenizer, structure_model_kwargs, pretrained_structure_path=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self._init_structure_model(
            structure_model_kwargs,
            pretrained_structure_path
        )

        self.structure_loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id
        )
    
    def _init_structure_model(self, structure_kwargs, pretrained_path):
        """Initialize or load the structure transformer model"""
        if pretrained_path is not None:
            print(f"Loading pretrained structure model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            ckpt_kwargs = checkpoint['hyper_parameters'].model.structure_model

            self.structure_model = TransformerModel(
                vocab_size = self.vocab_size,
                pad_token = self.tokenizer.pad_token_id,
                **ckpt_kwargs   
            )

            #Load weights
            structure_dict = {}
            for key, valye in checkpoint['state_dict'].items():
                if key.startswith('model.structure_model'):
                    new_key = key.replace('model.structure_model.', '')
                    structure_dict[new_key] = valye
            
            missing, unexpected = self.structure_model.load_state_dict(structure_dict, strict=False)

            if missing or unexpected:
                print(f"Warning: Missing={len(missing)}, Unexpected={len(unexpected)}")
            print("✅ Loaded structure model from checkpoint")
        
        else:
            # Initialize from scratch
            self.structure_model = TransformerModel(
                vocab_size=self.vocab_size,
                pad_token=self.tokenizer.pad_token_id,
                **structure_kwargs
            )
            self.structure_model.initialize_weights()
            print("✅ Initialized structure model from scratch") 


    def encode(self, batch):
        raise NotImplementedError()
    
    def forward(self, batch):
        raise NotImplementedError()

    def compute_structure_loss(self, outputs, batch):
        """Shared structure prediction loss computation"""
        target = batch['target'].view(-1)
        logits = outputs['structure_logits'].view(-1, outputs['structure_logits'].size(-1))
        return self.structure_loss_fn(logits, target)
    
    @torch.no_grad()
    def generate(
        self,
        structure_features,
        structure_features_mask,
        max_length=76,
        temperature=1.0,
        top_k=5,
        num_sequences=1
    ):

        encoder_features = structure_features
        batch_size       = encoder_features.size(0)
        device           = encoder_features.device

        expanded_encoder_features     = encoder_features.repeat_interleave(num_sequences, dim=0)
        expanded_batch_size           = batch_size * num_sequences
        expanded_encoder_padding_mask = structure_features_mask.repeat_interleave(num_sequences, dim=0)

        # Initialize output with start token
        decoder_input = torch.full(
            (expanded_batch_size, 1), 
            self.tokenizer.start_token_id, 
            dtype=torch.long, 
            device=device
        )

        sequence_neg_log_probs = torch.zeros(expanded_batch_size, dtype=torch.float, device=device)
        sequence_lengths       = torch.zeros(expanded_batch_size, dtype=torch.long, device=device)
        finished_sequences     = torch.zeros(expanded_batch_size, dtype=torch.bool, device=device)

        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Forward pass with current decoder input
            logits = self.structure_model(
                expanded_encoder_features,
                decoder_input,
                src_key_padding_mask=expanded_encoder_padding_mask
            )

            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / temperature
            full_probs        = F.softmax(next_token_logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(full_probs, top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            next_token_idx = torch.multinomial(top_k_probs, 1)
            selected_probs = torch.gather(top_k_probs, -1, next_token_idx)
            next_token     = torch.gather(top_k_indices, -1, next_token_idx)


            # For finished sequences, force padding tokens
            next_token[finished_sequences] = self.tokenizer.pad_token_id
            selected_probs[finished_sequences] = 1.0  # Don't penalize padding tokens
            not_finished = ~finished_sequences

            if not_finished.any():
                log_probs = torch.log(selected_probs[not_finished].view(-1) + 1e-8)
                log_probs = torch.clamp(log_probs, min=-100.0)
                sequence_neg_log_probs[not_finished] -= log_probs
            
            sequence_lengths[not_finished & (next_token.squeeze(-1) != self.tokenizer.eos_token_id)] += 1
            
            decoder_input = torch.cat((decoder_input, next_token), dim=1)

            eos_mask = (next_token.squeeze(-1) == self.tokenizer.eos_token_id) & ~finished_sequences
            finished_sequences = finished_sequences | eos_mask

            # Check if all sequences are finished
            if finished_sequences.all():
                break
        
        # Normalize scores by length
        sequence_lengths  = torch.clamp(sequence_lengths, min=1)
        normalized_scores = sequence_neg_log_probs / sequence_lengths.float()
        
        # Pad sequences to max_length if needed
        if decoder_input.size(1) < max_length:
            padding = torch.full(
                (expanded_batch_size, max_length - decoder_input.size(1)),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device
            )
            decoder_input = torch.cat((decoder_input, padding), dim=1)
        else:
            decoder_input = decoder_input[:, :max_length]
        
        # Reshape output
        all_sequences = decoder_input.view(batch_size, num_sequences, max_length)
        all_scores    = normalized_scores.view(batch_size, num_sequences)
        
        return all_sequences, all_scores
    
    @torch.no_grad()
    def evaluate_generation(self,
                            structure_features,
                            structure_features_mask,
                            true_smiles,
                            temperature=1.0,
                            top_k=5,
                            num_sequences=15,
                            max_samples=None,
                            max_molecule_len=45):

        """Evaluate generation performance using various metrics"""
        batch_size = len(true_smiles)
        sample_size = min(max_samples, batch_size) if max_samples else batch_size
        sample_indices = list(range(sample_size))
        
        # Sample features and SMILES
        sampled_features = structure_features[sample_indices]
        sampled_mask = structure_features_mask[sample_indices]
        sample_true_smiles = [true_smiles[i] for i in sample_indices]

        # Initialize metrics
        metrics = {
            'valid_rate': 0.0,
            'tanimoto_similarity': 0.0,
            'rdkit_similarity': 0.0,
            'maccs_similarity': 0.0,
            'exact_match_rate': 0.0,
            'levenshtein_distance': 0.0,
            'mces_distance': 0.0,
            'top_1_accuracy': 0.0,
            'top_5_accuracy': 0.0,
            'top_10_accuracy': 0.0,
            'top_15_accuracy': 0.0
        }

        try:
            # Generate sequences
            generated_ids, neg_log_probs = self.generate(
                structure_features=sampled_features,
                structure_features_mask=sampled_mask,
                max_length=max_molecule_len,
                temperature=temperature,
                top_k=top_k,
                num_sequences=num_sequences
            ) 

            batch_size, num_seqs, seq_len = generated_ids.shape
            # Decode sequences
            generated_smiles_lists = []

            for i in range(batch_size):
                sample_ids =generated_ids[i]
                sample_scores = neg_log_probs[i]
                decoded_smiles = []

                for j, seq_ids in enumerate(sample_ids):
                    tokens = []
                    for token in seq_ids:
                        if token.item() == self.tokenizer.eos_token_id:
                            break
                        if token.item() in [self.tokenizer.pad_token_id, self.tokenizer.start_token_id]:
                            continue
                        tokens.append(token.item())
                    
                    try:
                        smiles = self.tokenizer.decode(tokens)
                        decoded_smiles.append((smiles, sample_scores[j].item()))
                    except Exception as e:
                        decoded_smiles.append(("", float('inf')))
                
                #Sort by score (lower = better)
                sorted_smiles = sorted(decoded_smiles, key=lambda x: x[1])
                generated_smiles_lists.append([s[0] for s in sorted_smiles])
            
            #Calculate metrics
            valid_count = 0
            tanimoto_scores = []
            rdkit_scores = []
            maccs_scores = []
            exact_match_count = 0
            levenshtein_distances = []
            mces_distances = []
            top_1_hits = top_5_hits = top_10_hits = top_15_hits = 0

            for gen_smiles_list, true_smiles in zip(generated_smiles_lists, sample_true_smiles):
                canon_gen_smiles = []
                for gen_smiles in gen_smiles_list:
                    canon_gen = canonicalize_smiles(gen_smiles)
                    if canon_gen and is_valid_smiles(canon_gen):
                        canon_gen_smiles.append(canon_gen)

                canon_true = canonicalize_smiles(true_smiles)

                if canon_gen_smiles:
                    valid_count+=1

                    best_tanimoto = 0
                    best_rdkit = 0
                    best_maccs = 0
                    best_levenshtein = float('inf')
                    best_mces = float('inf')
                    exact_match = False

                    for canon_gen in canon_gen_smiles:
                        if canon_gen == canon_true:
                            exact_match = True

                        best_tanimoto = max(best_tanimoto, calculate_tanimoto_similarity(canon_gen, canon_true, use_chirality=True))
                        best_rdkit = max(best_rdkit, calculate_rdkit_similarity(canon_gen, canon_true))
                        best_maccs = max(best_maccs, calculate_maccs_similarity(canon_gen, canon_true))
                        best_levenshtein = min(best_levenshtein, compute_levenshtein_distance(canon_gen, canon_true))
                        best_mces = min(best_mces, compute_mces_distance(canon_gen, canon_true))
                    
                    tanimoto_scores.append(best_tanimoto)
                    rdkit_scores.append(best_rdkit)
                    maccs_scores.append(best_maccs)
                    levenshtein_distances.append(best_levenshtein)
                    mces_distances.append(best_mces)

                    if exact_match:
                        exact_match_count +=1

                    #Top-k accuracy
                    if canon_true in canon_gen_smiles[:1]:
                        top_1_hits +=1
                    if canon_true in canon_gen_smiles[:5]:
                        top_5_hits +=1
                    if canon_true in canon_gen_smiles[:10]:
                        top_10_hits +=1
                    if canon_true in canon_gen_smiles[:15]:
                        top_15_hits +=1
            
            # Calculate overall metrics
            metrics['valid_rate'] = valid_count / sample_size
            metrics['exact_match_rate'] = exact_match_count / sample_size
            metrics['top_1_accuracy'] = top_1_hits / sample_size
            metrics['top_5_accuracy'] = top_5_hits / sample_size
            metrics['top_10_accuracy'] = top_10_hits / sample_size
            metrics['top_15_accuracy'] = top_15_hits / sample_size

            if tanimoto_scores:
                metrics['tanimoto_similarity'] = sum(tanimoto_scores) / len(tanimoto_scores)
            if rdkit_scores:
                metrics['rdkit_similarity'] = sum(rdkit_scores) / len(rdkit_scores)
            if maccs_scores:
                metrics['maccs_similarity'] = sum(maccs_scores) / len(maccs_scores)
            if levenshtein_distances:
                metrics['levenshtein_distance'] = sum(levenshtein_distances) / len(levenshtein_distances)
            if mces_distances:
                metrics['mces_distance'] = sum(mces_distances) / len(mces_distances)
            
            return metrics
            
        except Exception as e:
            print(f"Error during generation evaluation: {str(e)}")
            traceback.print_exc()
            raise e




