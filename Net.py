import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 这里用来定义神经网络的结构

# Edge端网络的结构
class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        print(f"EdgeNet - fc1 output: {x[0][:5]}")
        x = torch.relu(self.fc2(x))
        print(f"EdgeNet - fc2 output: {x[0][:5]}")
        x = torch.relu(self.fc3(x))
        print(f"EdgeNet - fc3 output: {x[0][:5]}")
        return x

class EdgeClient
# Server端网络的结构
class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.fc4(x))
        print(f"ServerNet - fc4 output: {x[0][:5]}")
        x = torch.relu(self.fc5(x))
        print(f"ServerNet - fc5 output: {x[0][:5]}")
        x = torch.relu(self.fc6(x))
        print(f"ServerNet - fc6 output: {x[0][:5]}")
        x = self.fc7(x)
        print(f"ServerNet - fc7 output: {x[0][:5]}")
        return x