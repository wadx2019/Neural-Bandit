import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from alg import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)

class ReplayBuffer:

    def __init__(self, d, capacity):
        self.buffer = {'context':np.zeros((capacity, d)), 'reward': np.zeros((capacity,1))}
        self.capacity = capacity
        self.size = 0
        self.pointer = 0


    def add(self, context, reward):
        self.buffer['context'][self.pointer] = context
        self.buffer['reward'][self.pointer] = reward
        self.size = min(self.size+1, self.capacity)
        self.pointer = (self.pointer+1)%self.capacity

    def sample(self, n):
        idx = np.random.randint(0,self.size,size=n)
        return self.buffer['context'][idx], self.buffer['reward'][idx]

class NeuralTS(RandomAlg):

    def __init__(self, d, K, beta=1, lamb=1, hidden_size=128, lr=3e-4, reg=0.000625):
        self.K = K
        self.T = 0
        self.reg = reg
        self.beta = beta
        self.net = Model(d, hidden_size, 1)
        self.hidden_size = hidden_size
        self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.numel = sum(w.numel() for w in self.net.parameters() if w.requires_grad)
        self.sigma_inv = lamb * np.eye(self.numel, dtype=np.float32)
        self.device = device

        self.theta0 = torch.cat(
            [w.flatten() for w in self.net.parameters() if w.requires_grad]
        )
        self.replay_buffer = ReplayBuffer(d, 10000)

    def take_action(self, context):
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        g = np.zeros((self.K, self.numel), dtype=np.float32)

        for k in range(self.K):
            g[k] = self.grad(context[k]).cpu().numpy()

        with torch.no_grad():
            p = norm.rvs(loc=self.net(context).cpu().numpy(), scale=self.beta * np.sqrt(
                np.matmul(np.matmul(g[:, None, :], self.sigma_inv), g[:, :, None])[:, 0, :]))

        action = np.argmax(p)
        return action

    def grad(self, x):
        y = self.net(x)
        self.optimizer.zero_grad()
        y.backward()
        return torch.cat(
            [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.net.parameters() if w.requires_grad]
        ).to(self.device)

    def update(self, context, action, reward):
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        self.sherman_morrison_update(self.grad(context[action, None]).cpu().numpy()[:, None])
        self.replay_buffer.add(context[action].cpu().numpy(), reward)
        self.T += 1
        self.train()

    def sherman_morrison_update(self, v):
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1 + v.T @ self.sigma_inv @ v)

    def train(self):
        if self.T > self.K and self.T % 1 == 0:
            for _ in range(2):
                x, y = self.replay_buffer.sample(64)
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
                y_hat = self.net(x)
                loss = F.mse_loss(y_hat, y)
                loss += self.reg * torch.norm(torch.cat(
                    [w.flatten() for w in self.net.parameters() if w.requires_grad]
                ) - self.theta0) ** 2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

