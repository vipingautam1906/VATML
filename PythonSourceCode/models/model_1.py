import torch
import torch.nn as nn
import torch.nn.functional as F

class SysNet(nn.Module):
    def __init__(self):
        super(SysNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=32, stride=16, padding=0, bias=True),
            nn.BatchNorm1d(3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=16, stride=4, padding=0, bias=True),
            nn.ReLU()
        )
         
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc_input = self.get_input_size()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input,24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12,2)
        )

    def get_input_size(self):
        out = self.flatten(self.conv2(self.conv1(torch.rand(1,1,1250))))
        return out.shape[1]

    def forward(self, x):
        x = x.squeeze(dim=3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.flatten(x) 
        x = self.fc_layers(x)
        return x
