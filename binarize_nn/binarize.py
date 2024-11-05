import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = 'cpu'
save = False
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

@torch.no_grad()
def accuracy(model, test_data):
    val_loader = DataLoader(test_data, batch_size, shuffle=True)
    n_total = test_data.__len__()
    n_correct = 0

    for x, label in val_loader:
        x = x.to(device)
        out = model(x)
        prediction = torch.argmax(out, dim=-1)
        n_correct += torch.sum(label.to(device) == prediction)

    return n_correct / n_total * 100.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(28*28, d_model)
        self.output_layer = nn.Linear(d_model, 10)

        self.hidden_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'linear': nn.Linear(d_model, d_model)
            }) for _ in range(n_layers - 2)
        ])

    def binarize(self, W, b, t):


    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.input_layer(x.reshape(bs, -1))

        for layer in self.hidden_layers:
            x = F.gelu(layer['norm'](x))
            x = layer['linear'](x)

        return self.output_layer(F.gelu(x))


model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
#model = torch.compile(model)

model.load_state_dict(torch.load('./model/2_layer_mlp.pt'))
#optimizer.load_state_dict(torch.load('./model/2_layer_mlp_adam.pt'))

with torch.no_grad():
    out = model(x.to(device))
    print(out)
    print(out.shape)


#%%


#%%
for n in range(n_epochs):
    print("Epoch: ", n)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)

    for x, label in train_loader:
        out = model(x.to(device))
        loss = loss_fn(out, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = accuracy(model, val_data)
    print(acc)

print("Training done :)")

