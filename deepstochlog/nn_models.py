import torch


class TrainableProbability(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.vars = torch.nn.Parameter(torch.ones([1, N]))

    def forward(self, X):
        shape = X.shape[0]
        p = torch.softmax(self.vars, dim=-1).squeeze(0)
        return [p for _ in range(shape)]
