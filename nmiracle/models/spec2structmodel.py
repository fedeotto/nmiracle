import torch
import torch.nn as nn
from nmiracle.models.components.multispectra_encoder import MultiSpectralEncoder, MeanPool
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