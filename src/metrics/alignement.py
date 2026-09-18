import torch




class INFONCEMetric:


    def __call__(self,solution,predicted_solutions):
        scores = solution @ predicted_solutions.t()
        prediction = torch.argmax(scores,dim=-1)
        labels = torch.arange(scores.size(0),device=scores.device)
        accuracy = (prediction == labels).float().mean()
        return accuracy