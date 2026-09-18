import torch



class MLMMetrics:

    def __call__(self,logits,labels):
        with torch.no_grad():
            # taking the indice of the maximum logit for each token 
            predictions  = torch.argmax(logits,dim=-1)
            # extract tokens indices that were masked 
            mask = labels != -100 
            # in case any token has been masked (since we mask randomly)
            if mask.sum() > 0 :
                # number of correct tokens
                correct = (predictions[mask] == labels[mask]).float().sum()
                # compare the true predicted with all masked tokens
                accuracy = correct / mask.sum()
            else :
                accuracy = torch.tensor(0.0)
        return accuracy
        