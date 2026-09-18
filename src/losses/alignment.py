from torch import nn
import torch.nn.functional as F
import torch



class INFONCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,predictions,projections):
        predictions = F.normalize(predictions,dim=-1)
        projections = F.normalize(projections,dim=-1)
        scores  = predictions @ projections.t()
        # original labels are basically the indice of each sample in originals
        # correct scores lays in the diagonal
        labels = torch.arange(
            predictions.shape[0]
        ).to(scores.device)

        loss = F.cross_entropy(scores,labels)
        return loss




