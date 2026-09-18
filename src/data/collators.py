import torch

from .maskings import (
    mlm_masking,
    mask_problems,
    mask_solutions
)

class CodeBEPADataCollator:

    def __init__(self,tokenizer,mlm_probability,max_length):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length


    def __call__(self,batch):
        sequences = []
        positions = []
        for problem,solution in batch : 
            # convert the sequence to token 
            problem_tokens  = self.tokenizer.tokenize(problem)
            solution_tokens  = self.tokenizer.tokenize(solution)
            # We remove 4 because they were reserved for CLS and SEP for each sequence of pair 
            max_tokens  = self.max_length - 4 
            problem_limit = int(max_tokens * 0.6)
            solution_limit = int(max_tokens * 0.4)
            # Apply Truncation
            
            # the problem tokens must not exceed 60% of the max_tokens
            if len(problem_tokens) > problem_limit : 
                problem_tokens = problem_tokens[:problem_limit]
            # the solution with 40% 
            if len(solution_tokens) > solution_limit : 
                solution_tokens = solution_tokens[:solution_limit]

            # now we convert tokens to ID and we add [CLS] and [SEP] roken id 
            problem_ids = self.tokenizer.convert_tokens_to_ids(problem_tokens)
            solution_ids = self.tokenizer.convert_tokens_to_ids(solution_tokens)
            # concatenate them 
            sequences_ids = [
                 self.tokenizer.cls_token_id,
                 *problem_ids,
                 self.tokenizer.sep_token_id,
                 self.tokenizer.cls_token_id,
                 *solution_ids,
                 self.tokenizer.sep_token_id
            ]

            # Positions will be needed for aligment masking especially but also helpful for mlm masking
            problem_len = len(problem_ids)
            solution_len = len(solution_ids)
            seq_position = {
                "first_cls_pos" : 0,
                "problem_start" : 1,
                "problem_end" : problem_len,
                "first_sep_pos" : problem_len + 1,
                "second_cls_pos" : problem_len + 2,
                "solution_start" : problem_len + 3,
                "solution_end" : problem_len + 2 + solution_len,
                "second_sep_pos" : problem_len + 3 + solution_len
            }

            sequences.append(sequences_ids)
            positions.append(seq_position)

            # Apply mlm masking 




        input_ids , attention_mask  = self._add_paddings(sequences,self.max_length)

        # Masking the sequcenses 
        mlm_maskes,masked_labels = mlm_masking(input_ids,positions,self.tokenizer,self.mlm_probability)
        masked_problems  = mask_problems(input_ids,positions,self.tokenizer)

        masked_solutions = mask_solutions(input_ids,positions,self.tokenizer)

        second_cls_positions = torch.tensor(
             [p["second_cls_pos"] for p in positions],
             device=input_ids.device
        )
        return {
             "attention_mask" : attention_mask,
             "mlm_input_ids"  : mlm_maskes,
             "mlm_labels" : masked_labels,
             "masked_problems" : masked_problems,
             "masked_solutions" : masked_solutions,
             "second_cls_pos" : second_cls_positions
        }


    def _add_paddings(self,sequences_ids,max_length):
            batch_size = len(sequences_ids)
            # creating a batch full of padding ids 
            input_ids = torch.full((batch_size,max_length),self.tokenizer.pad_token_id,dtype=torch.long)
            # fill attention mask with zeros
            attention_mask = torch.zeros((batch_size,max_length),dtype=torch.long)

            # refill input ID's with their original input_ids and attention mask with 1 
            for i , seq in enumerate(sequences_ids):
                seq_len  = len(seq)
                input_ids[i,:seq_len] = torch.tensor(seq,dtype=torch.long)
                attention_mask[i,:seq_len] = 1

            return input_ids,attention_mask