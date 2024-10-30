import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

n_layers = 2
d_model = 1024
batch_size = 512
n_epochs = 25


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3015,))
])

train_data = datasets.MNIST(root='data', transform=transform, train=True)
val_data = datasets.MNIST(root='data', transform=transform, train=False)

def accuracy(model, test_data):
    val_loader = DataLoader(test_data, batch_size, shuffle=True)
    n_total = test_data.__len__()
    n_correct = 0

    with torch.no_grad():
        for x, label in val_loader:
            out = model(x)
            prediction = torch.argmax(out, dim=-1)
            n_correct += torch.sum(label == prediction)

    return n_correct / n_total * 100.


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(28*28, d_model)
        self.output_layer = nn.Linear(d_model, 10)
        self.hidden_layers = [
            (nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
            for _ in range(n_layers - 2)
        ]

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.input_layer(x.reshape(bs, -1))
        for layernorm, linear in self.hidden_layers:
            x = layernorm(F.gelu(x))
            x = linear(x)
        return self.output_layer(F.gelu(x))


model = Model()
#model = torch.compile(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

for n in range(n_epochs):
    print("Epoch: ", n)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)

    for x, label in train_loader:
        out = model(x)
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = accuracy(model, val_data)
    print(acc)

print("Training done :)")

torch.save(model.state_dict(), './model/2_layer_mlp.pt')
torch.save(optimizer.state_dict(), './model/2_layer_mlp_adam.pt')

print("Model saved ;)")
