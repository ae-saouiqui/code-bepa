
import torch
from torch.amp import GradScaler,autocast
from tqdm import tqdm
from src.metrics.mlm import MLMMetrics
from src.metrics.alignement import INFONCEMetric
from src.metrics.rankme import RankMEMetric
from src.losses.alignment import INFONCELoss



class CodeBEPATrainer:

    def __init__(self,model,optimizer,train_dataloader,val_dataloader,alignement_weight,device,accumulation_step):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.alignement_weight = alignement_weight
        self.accumulation_step = accumulation_step
        self.epoch = 0
        self.alignement_loss = INFONCELoss()
        self.mlm_metric = MLMMetrics()
        self.alignment_metric = INFONCEMetric()
        self.rankme_metric = RankMEMetric()
        self.scaler = GradScaler(self.device)

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)


    def train_epoch(self):
        self.model.train()
        # clear old grdients 
        self.optimizer.zero_grad()

        total_mlm_loss = 0.0
        total_alignment_loss = 0.0
        total_loss  = 0.0 
        total_mlm_accuracy = 0.0
        total_infonce_accuracy = 0.0
        num_batches = 0

        progress_bar  =  tqdm(self.train_loader,desc=f"Epoch {self.epoch} ",unit=" batch")
        for i,batch in enumerate(progress_bar):
            num_batches += 1 
            # moving tensor to corrending device 
            batch = {
                key : value.to(self.device) if torch.is_tensor(value) else value  for key,value in batch.items()
            }


            with autocast(self.device.type):
                # forward pass 
                mlm_output,_,z_solution,preicted_solution = self.model(batch)
                mlm_loss = mlm_output['loss']
                algn_loss = self.alignement_loss(preicted_solution , z_solution)
                total_loss_value = mlm_loss + self.alignement_weight * algn_loss
                loss =  total_loss_value / self.accumulation_step
                mlm_accuracy = self.mlm_metric(mlm_output['logits'],batch['mlm_labels'])
                infonce_accuracy = self.alignment_metric(z_solution,preicted_solution)
        
                
            
            self.scaler.scale(loss).backward()

            total_mlm_loss += mlm_loss.item()
            total_alignment_loss += algn_loss.item()
            total_loss += total_loss_value.item()
            total_mlm_accuracy += mlm_accuracy.item()
            total_infonce_accuracy += infonce_accuracy.item()

            if (i + 1) % self.accumulation_step == 0 :
                self._optimizer_step()

        return {
                "mlm_loss" : total_mlm_loss / num_batches,
                "alignement_loss" : total_alignment_loss  / num_batches,
                "total_loss" : total_loss / num_batches,
                "mlm_accuracy" : total_mlm_accuracy / num_batches,
                "infonce_accuracy" : total_infonce_accuracy / num_batches
        }

    def train(self,epochs : int):
        history = []
        for epoch in range(epochs) :
            self.epoch = epoch + 1
            train_result = self.train_epoch()
            print(train_result)
            val_result = self.validate()
            print(val_result)
            history.append(
                {
                    "epoch" : epoch + 1,
                    "train": train_result,
                    "val" : val_result
                }
            )
        return history
    
    def validate(self):
        # set evaluation mode 
        self.model.eval()

        total_mlm_loss = 0.0
        total_alignment_loss = 0.0
        total_loss  = 0.0 
        total_mlm_accuracy = 0.0
        total_infonce_accuracy = 0.0
        problem_embeddings = []
        solution_embeddings = []

        progress_bar = tqdm(self.val_loader,desc="Validation : ",unit=" batch")

        num_batches = 0

        with torch.inference_mode():
            for batch in progress_bar:
                num_batches += 1
                # moving tensor to corrending device 
                batch = {
                    key : value.to(self.device) if torch.is_tensor(value) else value  for key,value in batch.items()
                    }

                mlm_output,z_problem,z_solution,predicted_solution = self.model(batch)
                mlm_loss = mlm_output['loss']
                mlm_logits = mlm_output['logits']
                algn_loss = self.alignement_loss(predicted_solution,z_solution)
                loss = mlm_loss + self.alignement_weight * algn_loss
                mlm_accuracy = self.mlm_metric(mlm_logits,batch["mlm_labels"])
                alignement_accuracy = self.alignment_metric(z_solution,predicted_solution)
                total_mlm_loss += mlm_loss.item()
                total_alignment_loss += algn_loss.item()
                total_loss += loss.item()
                total_mlm_accuracy += mlm_accuracy.item() 
                total_infonce_accuracy += alignement_accuracy.item()

                problem_embeddings.append(z_problem.cpu())
                solution_embeddings.append(z_solution.cpu())

            problem_embeddings = torch.cat(problem_embeddings, dim=0)
            solution_embeddings = torch.cat(solution_embeddings, dim=0)
            problem_rankme = self.rankme_metric(problem_embeddings)
            solution_rankme = self.rankme_metric(solution_embeddings)

            return {
                "mlm_loss" : total_mlm_loss / num_batches,
                "alignement_loss" : total_alignment_loss  / num_batches,
                "total_loss" : total_loss / num_batches,
                "mlm_accuracy" : total_mlm_accuracy / num_batches,
                "infonce_accuracy" : total_infonce_accuracy / num_batches,
                "problem_rankme" : problem_rankme,
                "solution_rankme" : solution_rankme
                }

    def save_checkpoint(self, folder):
        model = (
            self.model._orig_mod
            if hasattr(self.model, "_orig_mod")
            else self.model
            )

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict()
            }

        torch.save(
            checkpoint,
            f"{folder}/epoch_{self.epoch}.pt"
            )