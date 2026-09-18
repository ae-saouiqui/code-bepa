import torch
from transformers import AutoTokenizer
from src.training.trainer import CodeBEPATrainer
import json 
from torch.utils.data import DataLoader
from src.data.dataset import CodeBEPADataSet
from src.data.collators import CodeBEPADataCollator
from src.schemas.trainer_arguments import CodeBEPATrainingArgs
from torch.optim import AdamW
from src.models.bepa import CodeBEPA
import json

def main():
    with open("src/configs/train.json","r") as file :
        configs  = json.load(file)

    # create dataset 
    train_dataset = CodeBEPADataSet(configs["train_dataset"])
    val_dataset = CodeBEPADataSet(configs["val_dataset"])
    # Getting a tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])
    fn_collator = CodeBEPADataCollator(tokenizer,configs["mlm_probability"],configs["max_length"])
    # set dataloader 
    train_dataloader = DataLoader(train_dataset,configs["batch_size"],drop_last=True,collate_fn=fn_collator,pin_memory=configs["pin_memory"],num_workers=configs["num_workers"])
    val_dataloader = DataLoader(val_dataset,configs["batch_size"],drop_last=True,collate_fn=fn_collator,pin_memory=configs["pin_memory"],num_workers=configs["num_workers"])
    # Initiating a model
    model = CodeBEPA(configs["model_name"],configs["hidden_size"],configs["output_dim"])

    # Compiling the model 
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(),lr=configs["learning_rate"])

    training_args = CodeBEPATrainingArgs(
        model = model,
        optimizer = optimizer,
        train_dataloader= train_dataloader,
        val_dataloader = val_dataloader,
        alignement_weight= configs["alignement_weight"],
        accumulation_step = configs["accumulation_step"],
        device=configs["device"]
        )._asdict()

    trainer = CodeBEPATrainer(**training_args)
    history   = trainer.train(configs["epochs"])

    with open(f"results/{configs["model_name"]}.json","w") as file :
        json.dump(history,file)
        
    trainer.save_checkpoint(configs["save_folder"])


if __name__ == "__main__":
    main()


