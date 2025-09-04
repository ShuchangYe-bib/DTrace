import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PositionwiseFeedForward(nn.Module):
	
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class ClassificationHead(nn.Module):

    def __init__(self, d_model, num_classes, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        x = self.activation(x)
        return x



