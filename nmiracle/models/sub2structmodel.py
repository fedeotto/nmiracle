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
from nmiracle.models.base_model import BaseModel
import torch.nn.functional as F
import traceback

class Sub2Structure(BaseModel):
    def __init__(
        self,
        tokenizer,
        structure_kwargs,
        pretrained_structure_path=None,
        num_substructures=991,
        max_substructures_count=232,
    ):
        super().__init__(
            tokenizer=tokenizer,
            structure_model_kwargs=structure_kwargs,
            pretrained_structure_path=pretrained_structure_path
        )

        self.num_substructures = num_substructures
        self.max_substructures_count = max_substructures_count

        self.substructure_embedding = nn.Embedding(
            num_embeddings=num_substructures + 1, 
            embedding_dim=structure_kwargs['d_model'],
            padding_idx=0
        )
        
        self.substructure_count_embedding = nn.Embedding(
            num_embeddings=max_substructures_count + 1,
            embedding_dim=structure_kwargs['d_model'],
            padding_idx=0
        )

        self.mlp_proj = nn.Sequential(
            nn.Linear(structure_kwargs['d_model'], structure_kwargs['d_model']),
            nn.GELU(),
            nn.LayerNorm(structure_kwargs['d_model'], eps=structure_kwargs['layer_norm_eps']),
        )
    
    def encode(self, batch):
        """Encode substructures into features"""
        # Get inputs
        substructures =  batch['substructures']
        src_key_pad_mask = (substructures == 0)  
        structure_features = self.substructure_embedding(substructures)  # [batch_size, seq_len, d_model]

        device_type = 'cuda' if structure_features.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            encoded_features = self.structure_model.transformer.encoder(
                structure_features,
                src_key_padding_mask=src_key_pad_mask
            )

        return encoded_features, src_key_pad_mask


    def forward(self, batch):
        """Full forward pass with decoder"""
        substructures = batch['substructures']
        substructure_counts = batch['substructure_counts']

        src_key_pad_mask = (substructures == 0)
        substructure_features = self.substructure_embedding(substructures)  
        count_features = self.substructure_count_embedding(substructure_counts)

        # Combine embeddings (element-wise addition)
        combined_features = substructure_features + count_features    
        projected_features = self.mlp_proj(combined_features)
        
        # Get decoder inputs for teacher forcing
        decoder_input = batch['decoder_input']
        
        # Forward with teacher forcing using pre-created decoder input
        structure_logits = self.structure_model(
            projected_features,
            decoder_input,
            src_key_padding_mask=src_key_pad_mask
        )
        
        return {
            'structure_features': projected_features,
            'structure_features_mask': src_key_pad_mask,
            'structure_logits': structure_logits
        }

    def compute_loss(self, outputs, batch):
        structure_loss = self.compute_structure_loss(outputs, batch)
        return structure_loss, {'structure_loss': structure_loss.item()}
