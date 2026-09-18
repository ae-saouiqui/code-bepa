from torch import nn

class Projector(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,output_dim)
        )

    def forward(self,x):
        projection = self.mlp(x)
        return projection
