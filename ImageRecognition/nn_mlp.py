import torch
from torch import nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def init(self):
        super(MLPNetwork, self).init()
        self.fc1 = nn.Linear(2828, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 2828)   # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)