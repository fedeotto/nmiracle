from rdkit import Chem
import selfies as sf

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def smiles_to_selfies(smiles_string):
    """Convert SMILES to SELFIES format"""
    try:
        return sf.encoder(smiles_string)
    except:
        return "[]"  # Empty SELFIES as fallback

def selfies_to_smiles(selfies_string):
    """Convert SELFIES to SMILES format"""
    try:
        return sf.decoder(selfies_string)
    except:
        return "C"  # Fallback to methane

def convert_for_model(smiles_string, model_format):
    """Convert SMILES to the format needed by model"""
    if model_format.lower() == "smiles":
        return smiles_string
    elif model_format.lower() == "selfies":
        return smiles_to_selfies(smiles_string)
    return smiles_string  # Default to SMILES

def convert_to_smiles(molecule_string, model_format):
    """Convert from model format back to SMILES for evaluation"""
    if model_format.lower() == "smiles":
        return molecule_string
    elif model_format.lower() == "selfies":
        return selfies_to_smiles(molecule_string)
    return molecule_string  # Default to assuming it's already SMILES