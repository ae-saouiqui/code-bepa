import torch
from torch import nn
from transformers import AutoModelForMaskedLM
from .projector import Projector



class CodeBEPA(nn.Module):

    def __init__(self,model_name,hidden_size,output_dim):
        super().__init__()
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        # Use the base RoBERTa encoder without the MLM head (Just to remember : Check the source RobertForMaskedLM)
        self.base_model = self.mlm_model.roberta
        self.projector = Projector(self.base_model.config.hidden_size,hidden_size,output_dim)
        self.predictor = nn.Linear(output_dim,output_dim,bias=False)

    def forward(self,batch):
        batch_size = len(batch["mlm_input_ids"])
        mlm_input = {"input_ids": batch["mlm_input_ids"],"attention_mask":batch["attention_mask"],"labels":batch["mlm_labels"]}
        bepa_input  = {
            "input_ids" : torch.cat(
                [batch["masked_solutions"],batch["masked_problems"]],
                dim=0
            ),
            "attention_mask" : torch.cat(
                [batch["attention_mask"],batch["attention_mask"]]  ,
                dim=0,
            )
        }
        # problem_input  = {"input_ids":batch["masked_solutions"], "attention_mask" : batch["attention_mask"]}
        # solution_input  = {"input_ids":batch["masked_problems"], "attention_mask" : batch["attention_mask"]}


        output = self.mlm_model(**mlm_input)
        mlm_output = {
            "loss" : output.loss,
            "logits" : output.logits
        }

        bepa_output = self.base_model(**bepa_input)

        x_problem,x_solution = bepa_output.last_hidden_state.chunk(2,dim=0)
        # x_problem = self.base_model(**problem_input).last_hidden_state
        cls_problems = x_problem[:,0,:]

        # x_solution = self.base_model(**solution_input).last_hidden_state

        # cls_sol_positions = torch.tensor(
        #     [p["second_cls_pos"] for p in batch["positions"]],
        #     device=x_solution.device
        #     )
        
        cls_sol_positions = batch["second_cls_pos"]
        
        batch_indices = torch.arange(batch_size,device=x_solution.device)
        cls_solutions = x_solution[batch_indices,cls_sol_positions]

        z_problem = self.projector(cls_problems)
        z_solution = self.projector(cls_solutions)

        # predicting the last second [CLS] 
        predicted_solution = self.predictor(z_problem)
        return mlm_output,z_problem,z_solution,predicted_solution

    def predict(self,problem):
        output = self.base_model(**problem)
        cls_problem  = output.last_hidden_state[:,0,:]
        z_problem = self.projector(cls_problem)
        predicted_solution = self.predictor(z_problem)
        return predicted_solution


        
