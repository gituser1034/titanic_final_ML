import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        # Makes logit
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class MLPClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear1 = nn.Linear(num_features, 100)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(100, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.output(x)
        return x


class MLP2DClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear1 = nn.Linear(num_features, 100)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x


class MLP4DResidualClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.act = nn.ReLU()
        self.num_features = num_features

        # Main layers
        self.linear1 = nn.Linear(num_features, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)

        # Skip connections
        self.skip1 = nn.Linear(num_features, 128)
        self.skip2 = nn.Linear(128, 64)
        self.skip3 = nn.Linear(64, 32)
        self.skip4 = nn.Linear(32, 16)

        self.output = nn.Linear(16, 2)

    def forward(self, x):
        s1 = self.skip1(x)
        x = self.act(self.linear1(x) + s1)

        s2 = self.skip2(x)
        x = self.act(self.linear2(x) + s2)

        s3 = self.skip3(x)
        x = self.act(self.linear3(x) + s3)

        s4 = self.skip4(x)
        x = self.act(self.linear4(x) + s4)

        return self.output(x)


class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

