import numpy as np
import torch
from transformers import TokenizersBackend


def mlm_masking(sequences,positions,tokenizer:TokenizersBackend,pourcentage:float=0.15):
    if not isinstance(pourcentage,float) :
        raise TypeError("'pourcentage has to be a float'")
    if not  (0. <= pourcentage <= 1.) : 
        raise ValueError("'pourcentage' must be in range  [0,1]")

    input_ids = sequences.clone()
    # the unmasked tokens will have an id -100 while masked will habe their real id
    labels = torch.full_like(input_ids,-100)
    for i,seq in enumerate(input_ids) : 
        candidate_positions  = []
        candidate_positions.extend(range(positions[i]["problem_start"],positions[i]["problem_end"] + 1))
        candidate_positions.extend(range(positions[i]["solution_start"],positions[i]["solution_end"] + 1))

        # Deciding for each position wether to mask that position or not 
        # the probability for a position to be masked is 0.15, so 
        masked_decisions = np.random.binomial(1,pourcentage,len(candidate_positions))

        for j,is_masked in enumerate(masked_decisions):
            if is_masked : 
                candidate_poisition  = candidate_positions[j]
                original_token = seq[candidate_poisition].clone()
                p = np.random.random()
                if p < 0.8 : 
                    seq[candidate_poisition] = tokenizer.mask_token_id
                elif p < 0.9 : 
                    n = np.random.randint(0,tokenizer.vocab_size)
                    seq[candidate_poisition] = n

                labels[i,candidate_poisition] = original_token
    return input_ids,labels


def mask_problems(seuqences,positions,tokenizer):
    inputs_ids = seuqences.clone()

    for i,seq in enumerate(inputs_ids):
        problem_end = positions[i]["first_sep_pos"] + 1
        # BERT architecture expects the first token to be [CLS] that is why we unmask it
        problem_start = positions[i]["problem_start"] 
        seq[problem_start:problem_end ] = tokenizer.mask_token_id

    return inputs_ids

def mask_solutions(sequences,positions,tokenizer):
    inputs_ids = sequences.clone()

    for i,seq in enumerate(inputs_ids):
        solution_start = positions[i]["second_cls_pos"]
        solution_end = positions[i]["second_sep_pos"] +  1 
        seq[solution_start:solution_end] = tokenizer.mask_token_id

    return inputs_ids



    
    
    
