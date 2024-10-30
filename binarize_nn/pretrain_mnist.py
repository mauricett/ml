import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


n_layers = 24
d_model = 1024
batch_size = 1024

mnist_args = {
    'root': 'data',
    'train': True,
    'download': False
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3015,))
])

train_data = datasets.MNIST(**mnist_args, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer  = nn.Linear(28*28, d_model)
        self.output_layer = nn.Linear(d_model, 10)
        self.hidden_layers = [
            (nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
            for _ in range(n_layers - 2)
        ]

    def forward(self, x):
        x = self.input_layer(x.reshape(batch_size, -1))
        for layernorm, linear in self.hidden_layers:
            x = layernorm(F.gelu(x))
            x = linear(x)
        return self.output_layer(F.gelu(x))


x, y = next(iter(train_loader))
model = Model()
loss_fn = nn.CrossEntropyLoss()

#%%
with torch.no_grad():
    out = model(x)

loss = loss_fn(out, y)
