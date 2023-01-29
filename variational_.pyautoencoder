#// Geovani Rodriguez //#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST('./data', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist, batch_size=128, shuffle=True, num_workers=0)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, z_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Train the VAE model
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.view(-1, 28 * 28), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch: {} Train Loss: {:.6f}'.format(epoch, train_loss / len(train_loader)))

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize the model, optimizer, and optimiezer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Running the training loop for a specified number of epochs
for epoch in range(1, epochs + 1):
    train(epoch)
