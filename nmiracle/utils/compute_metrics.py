"""
This script computes ALL metrics with three levels of stereochemistry handling:
1. Exact match (strict)
2. Enantiomer-tolerant match (appropriate for NMR)
3. Constitution-only match (no stereochemistry)
"""
import json
from tqdm import tqdm
from common.eval_utils import (compute_mces_distance, 
                               calculate_tanimoto_similarity, 
                               calculate_rdkit_similarity, calculate_maccs_similarity, 
                               are_enantiomers,
                               compute_levenshtein_distance)
from rdkit import Chem
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

RESULTS_PATH = 'nmiracle/ckpts/nmiracle_spec2struct_ir_hnmr_cnmr/'
RESULTS_NAME = 'test_results-temperature-1.0-top_k-5.json'

def main():
    # Load results
    print(f"Loading results from {RESULTS_PATH}")
    with open(os.path.join(RESULTS_PATH, RESULTS_NAME), 'r') as f:
        results = json.load(f)

    all_canon_true = results['generations']['all_canon_true']
    all_canon_generated = results['generations']['all_canon_generated']
    total_molecules = len(all_canon_true)
    print(f"Processing {total_molecules} molecules...")

    # Level 1: EXACT MATCH (with stereochemistry)
    exact_top_1 = exact_top_5 =  exact_top_10 = exact_top_15 = exact_match_count = 0
    exact_tanimoto = []
    exact_rdkit = []
    exact_maccs = []
    exact_levenshtein = []
    exact_mces = []

    # Level 2: ENANTIOMER-AWARE
    enantiomer_top_1 = enantiomer_top_5 = enantiomer_top_10 = enantiomer_top_15 = enantiomer_match_count = 0
    enantiomer_tanimoto = []
    enantiomer_rdkit = []
    enantiomer_maccs = []
    enantiomer_levenshtein = []
    enantiomer_mces = []

    # Level 3: CONSTITUTIONAL (no stereochemistry)
    constitution_top_1 = constitution_top_5 = constitution_top_10 = constitution_top_15 = constitution_match_count = 0
    constitution_tanimoto = []
    constitution_rdkit = []
    constitution_maccs = []
    constitution_levenshtein = []
    constitution_mces = []

    valid_count = 0

    # Process each molecule
    for i, (gen_smiles_list, true_smiles) in enumerate(tqdm(zip(all_canon_generated, all_canon_true), 
                                                        total=total_molecules,
                                                        desc="Computing metrics")):

        
        # Check if we have valid predictions
        if not gen_smiles_list:
            continue
        
        valid_count += 1
        
        # Get canonical versions for comparison
        mol_true = Chem.MolFromSmiles(true_smiles)
        can_true_with_stereo = Chem.MolToSmiles(mol_true)
        can_true_no_stereo = Chem.MolToSmiles(mol_true, isomericSmiles=False)
        
        # Track best similarities for each level
        best_exact = {'tanimoto': 0, 'rdkit': 0, 'maccs': 0, 'levenshtein': float('inf'), 'mces': float('inf')}
        best_enantiomer = {'tanimoto': 0, 'rdkit': 0, 'maccs': 0, 'levenshtein': float('inf'), 'mces': float('inf')}
        best_constitution = {'tanimoto': 0, 'rdkit': 0, 'maccs': 0, 'levenshtein': float('inf'), 'mces': float('inf')}
        
        # Track if we found matches and at what rank
        found_exact = found_enantiomer = found_constitution = False
        exact_found_at = enantiomer_found_at = constitution_found_at = -1
        
        # Check each generated molecule
        for rank, pred_smiles in enumerate(gen_smiles_list):
            mol_pred = Chem.MolFromSmiles(pred_smiles)
            if mol_pred is None:
                continue
            
            can_pred_with_stereo = Chem.MolToSmiles(mol_pred)
            can_pred_no_stereo = Chem.MolToSmiles(mol_pred, isomericSmiles=False)
            
            # Check matches at three levels
            exact_match = (can_pred_with_stereo == can_true_with_stereo)
            enantiomer_match = exact_match or are_enantiomers(can_pred_with_stereo, can_true_with_stereo)
            constitution_match = (can_pred_no_stereo == can_true_no_stereo)
            
            # Track first occurrence for top-k accuracy
            if exact_match and not found_exact:
                found_exact = True
                exact_found_at = rank
            if enantiomer_match and not found_enantiomer:
                found_enantiomer = True
                enantiomer_found_at = rank
            if constitution_match and not found_constitution:
                found_constitution = True
                constitution_found_at = rank
            
            # === LEVEL 1: Compute similarities with stereochemistry (exact) ===
            tanimoto = calculate_tanimoto_similarity(can_pred_with_stereo, can_true_with_stereo, use_chirality=True)
            rdkit_sim = calculate_rdkit_similarity(can_pred_with_stereo, can_true_with_stereo)
            maccs_sim = calculate_maccs_similarity(can_pred_with_stereo, can_true_with_stereo)
            levenshtein = compute_levenshtein_distance(can_pred_with_stereo, can_true_with_stereo)
            mces_dist = compute_mces_distance(can_pred_with_stereo, can_true_with_stereo)
            
            if tanimoto > best_exact['tanimoto']:
                best_exact['tanimoto'] = tanimoto
            if rdkit_sim > best_exact['rdkit']:
                best_exact['rdkit'] = rdkit_sim
            if maccs_sim > best_exact['maccs']:
                best_exact['maccs'] = maccs_sim
            if levenshtein < best_exact['levenshtein']:
                best_exact['levenshtein'] = levenshtein
            if mces_dist < best_exact['mces']:
                best_exact['mces'] = mces_dist

            # === LEVEL 2: For enantiomer-aware, use same similarities ===
            # (Enantiomers have same fingerprints, so similarities are identical)
            if tanimoto > best_enantiomer['tanimoto']:
                best_enantiomer['tanimoto'] = tanimoto
            if rdkit_sim > best_enantiomer['rdkit']:
                best_enantiomer['rdkit'] = rdkit_sim
            if maccs_sim > best_enantiomer['maccs']:
                best_enantiomer['maccs'] = maccs_sim
            if levenshtein < best_enantiomer['levenshtein']:
                best_enantiomer['levenshtein'] = levenshtein
            if mces_dist < best_enantiomer['mces']:
                best_enantiomer['mces'] = mces_dist

            # === LEVEL 3: Compute similarities without stereochemistry ===
            tanimoto_const = calculate_tanimoto_similarity(can_pred_no_stereo, can_true_no_stereo, use_chirality=False)
            rdkit_const = rdkit_sim #calculate_rdkit_similarity(can_pred_no_stereo, can_true_no_stereo) (these ignore stereo)
            maccs_const = maccs_sim #calculate_maccs_similarity(can_pred_no_stereo, can_true_no_stereo)
            levenshtein_const = compute_levenshtein_distance(can_pred_no_stereo, can_true_no_stereo)
            mces_const = mces_dist#compute_mces_distance(can_pred_no_stereo, can_true_no_stereo)

            if tanimoto_const > best_constitution['tanimoto']:
                best_constitution['tanimoto'] = tanimoto_const
            if rdkit_const > best_constitution['rdkit']:
                best_constitution['rdkit'] = rdkit_const
            if maccs_const > best_constitution['maccs']:
                best_constitution['maccs'] = maccs_const
            if levenshtein_const < best_constitution['levenshtein']:
                best_constitution['levenshtein'] = levenshtein_const
            if mces_const < best_constitution['mces']:
                best_constitution['mces'] = mces_const
        
        # Store best similarities for this molecule
        exact_tanimoto.append(best_exact['tanimoto'])
        exact_rdkit.append(best_exact['rdkit'])
        exact_maccs.append(best_exact['maccs'])
        exact_levenshtein.append(best_exact['levenshtein'])
        exact_mces.append(best_exact['mces'])
        
        enantiomer_tanimoto.append(best_enantiomer['tanimoto'])
        enantiomer_rdkit.append(best_enantiomer['rdkit'])
        enantiomer_maccs.append(best_enantiomer['maccs'])
        enantiomer_levenshtein.append(best_enantiomer['levenshtein'])
        enantiomer_mces.append(best_enantiomer['mces'])
        
        constitution_tanimoto.append(best_constitution['tanimoto'])
        constitution_rdkit.append(best_constitution['rdkit'])
        constitution_maccs.append(best_constitution['maccs'])
        constitution_levenshtein.append(best_constitution['levenshtein'])
        constitution_mces.append(best_constitution['mces'])
        
        # Update match counters
        if found_exact:
            exact_match_count += 1
            if exact_found_at < 1:
                exact_top_1 += 1
            if exact_found_at < 5:
                exact_top_5 += 1
            if exact_found_at < 10:
                exact_top_10 += 1
            if exact_found_at < 15:
                exact_top_15 += 1
        
        if found_enantiomer:
            enantiomer_match_count += 1
            if enantiomer_found_at < 1:
                enantiomer_top_1 += 1
            if enantiomer_found_at < 5:
                enantiomer_top_5 += 1
            if enantiomer_found_at < 10:
                enantiomer_top_10 += 1
            if enantiomer_found_at < 15:
                enantiomer_top_15 += 1
        
        if found_constitution:
            constitution_match_count += 1
            if constitution_found_at < 1:
                constitution_top_1 += 1
            if constitution_found_at < 5:
                constitution_top_5 += 1
            if constitution_found_at < 10:
                constitution_top_10 += 1
            if constitution_found_at < 15:
                constitution_top_15 += 1

    print("\n" + "="*80)
    print("EVALUATION RESULTS - ALL METRICS")
    print("="*80)

    print("\n1. EXACT MATCH (Strict - all stereochemistry must match)")
    print("-" * 80)
    print(f"Exact match rate:      {exact_match_count/total_molecules:.4f}")
    print(f"Top-1 accuracy:        {exact_top_1/total_molecules:.4f}")
    print(f"Top-5 accuracy:        {exact_top_5/total_molecules:.4f}")
    print(f"Top-10 accuracy:       {exact_top_10/total_molecules:.4f}")
    print(f"Top-15 accuracy:       {exact_top_15/total_molecules/total_molecules:.4f}")
    print(f"Tanimoto similarity:   {sum(exact_tanimoto)/len(exact_tanimoto):.4f}")
    print(f"RDKit similarity:      {sum(exact_rdkit)/len(exact_rdkit):.4f}")
    print(f"MACCS similarity:      {sum(exact_maccs)/len(exact_maccs):.4f}")
    print(f"MCES similarity:       {sum(exact_mces)/len(exact_mces):.4f}")
    print(f"Levenshtein distance:  {sum(exact_levenshtein)/len(exact_levenshtein):.4f}")

    print("\n2. ENANTIOMER-AWARE MATCH (NMR-appropriate - treats enantiomers as equivalent)")
    print("-" * 80)
    print(f"Match rate:            {enantiomer_match_count/total_molecules:.4f}")
    print(f"Top-1 accuracy:        {enantiomer_top_1/total_molecules:.4f}")
    print(f"Top-5 accuracy:        {enantiomer_top_5/total_molecules:.4f}")
    print(f"Top-10 accuracy:       {enantiomer_top_10/total_molecules:.4f}")
    print(f"Top-15 accuracy:       {enantiomer_top_15/total_molecules:.4f}")
    print(f"Tanimoto similarity:   {sum(enantiomer_tanimoto)/len(enantiomer_tanimoto):.4f}")
    print(f"RDKit similarity:      {sum(enantiomer_rdkit)/len(enantiomer_rdkit):.4f}")
    print(f"MACCS similarity:      {sum(enantiomer_maccs)/len(enantiomer_maccs):.4f}")
    print(f"MCES similarity:       {sum(enantiomer_mces)/len(enantiomer_mces):.4f}")
    print(f"Levenshtein distance:  {sum(enantiomer_levenshtein)/len(enantiomer_levenshtein):.4f}")

    print("\n3. CONSTITUTIONAL MATCH (No stereochemistry)")
    print("-" * 80)
    print(f"Match rate:            {constitution_match_count/total_molecules:.4f}")
    print(f"Top-1 accuracy:        {constitution_top_1/total_molecules:.4f}")
    print(f"Top-5 accuracy:        {constitution_top_5/total_molecules:.4f}")
    print(f"Top-10 accuracy:       {constitution_top_10/total_molecules:.4f}")
    print(f"Top-15 accuracy:       {constitution_top_15/total_molecules:.4f}")
    print(f"Tanimoto similarity:   {sum(constitution_tanimoto)/len(constitution_tanimoto):.4f}")
    print(f"RDKit similarity:      {sum(constitution_rdkit)/len(constitution_rdkit):.4f}")
    print(f"MACCS similarity:      {sum(constitution_maccs)/len(constitution_maccs):.4f}")
    print(f"MCES similarity:       {sum(constitution_mces)/len(constitution_mces):.4f}")
    print(f"Levenshtein distance:  {sum(constitution_levenshtein)/len(constitution_levenshtein):.4f}")

    # Save results
    results_detailed = {
        'metrics': {
            'exact_match': {
                'exact_match_rate': exact_match_count/total_molecules,
                'top_1_accuracy': exact_top_1/total_molecules,
                'top_5_accuracy': exact_top_5/total_molecules,
                'top_10_accuracy': exact_top_10/total_molecules,
                'top_15_accuracy': exact_top_15/total_molecules,
                'tanimoto_similarity': sum(exact_tanimoto)/len(exact_tanimoto),
                'rdkit_similarity': sum(exact_rdkit)/len(exact_rdkit),
                'maccs_similarity': sum(exact_maccs)/len(exact_maccs),
                'mces_distance': sum(exact_mces)/len(exact_mces),
                'levenshtein_distance': sum(exact_levenshtein)/len(exact_levenshtein),
            },
            'enantiomer_aware': {
                'match_rate': enantiomer_match_count/total_molecules,
                'top_1_accuracy': enantiomer_top_1/total_molecules,
                'top_5_accuracy': enantiomer_top_5/total_molecules,
                'top_10_accuracy': enantiomer_top_10/total_molecules,
                'top_15_accuracy': enantiomer_top_15/total_molecules,
                'tanimoto_similarity': sum(enantiomer_tanimoto)/len(enantiomer_tanimoto),
                'rdkit_similarity': sum(enantiomer_rdkit)/len(enantiomer_rdkit),
                'maccs_similarity': sum(enantiomer_maccs)/len(enantiomer_maccs),
                'mces_distance': sum(enantiomer_mces)/len(enantiomer_mces),
                'levenshtein_distance': sum(enantiomer_levenshtein)/len(enantiomer_levenshtein),
            },
            'constitutional': {
                'match_rate': constitution_match_count/total_molecules,
                'top_1_accuracy': constitution_top_1/total_molecules,
                'top_5_accuracy': constitution_top_5/total_molecules,
                'top_10_accuracy': constitution_top_10/total_molecules,
                'top_15_accuracy': constitution_top_15/total_molecules,
                'tanimoto_similarity': sum(constitution_tanimoto)/len(constitution_tanimoto),
                'rdkit_similarity': sum(constitution_rdkit)/len(constitution_rdkit),
                'maccs_similarity': sum(constitution_maccs)/len(constitution_maccs),
                'mces_distance': sum(constitution_mces)/len(constitution_mces),
                'levenshtein_distance': sum(constitution_levenshtein)/len(constitution_levenshtein),
            }
        }
    }

    output_path = os.path.join(RESULTS_PATH, f'all_metrics_{RESULTS_NAME}')
    with open(output_path, 'w') as f:
        json.dump(results_detailed, f, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
