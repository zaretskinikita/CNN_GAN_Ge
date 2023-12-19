import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.seq_len = params.seq_len
        self.model = nn.Sequential(            
            nn.Conv1d(self.seq_len, self.seq_len // 2, 1,  bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.seq_len // 2, self.seq_len // 4, 1, bias=False),
            nn.BatchNorm1d(self.seq_len // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.seq_len // 4, self.seq_len // 8, 1, bias=False),
            nn.BatchNorm1d(self.seq_len // 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(self.seq_len // 8, self.seq_len // 16, 1, bias=False),
            nn.BatchNorm1d(self.seq_len // 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.seq_len // 16, 1, 1, bias=False),
        )
    def forward(self, x):
        x = x.unsqueeze(-1)
        output = self.model(x)
        return torch.sigmoid(output.squeeze(-1))

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.seq_len = params.seq_len
        self.random_dim = params.random_dim
        self.model = nn.Sequential(
            nn.ConvTranspose1d(self.random_dim, self.seq_len // 4, 1, bias=False),
            nn.BatchNorm1d(self.seq_len // 4),
            nn.ReLU(True),

            nn.ConvTranspose1d(self.seq_len // 4, self.seq_len // 2, 1, bias=False),
            nn.BatchNorm1d(self.seq_len // 2),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(self.seq_len // 2, self.seq_len, 1, bias=False)
      )
    def forward(self, z):
        z = z.unsqueeze(-1)
        output = self.model(z)
        return output.squeeze(-1)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.init_len = params.init_len
        self.emb_dim = params.seq_len
        self.model = nn.Sequential(            
            nn.Conv1d(self.init_len, self.init_len // 2, 1,  bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.init_len // 2, self.init_len // 4, 1, bias=False),
            nn.BatchNorm1d(self.init_len // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.init_len // 4, self.init_len // 8, 1, bias=False),
            nn.BatchNorm1d(self.init_len // 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.init_len // 8, self.emb_dim, 1, bias=False),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        output = self.model(x)
        return output.squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.init_len = params.init_len
        self.emb_dim = params.seq_len
        self.model = nn.Sequential(
            nn.ConvTranspose1d(self.emb_dim, self.init_len // 8, 1, bias=False),
            nn.BatchNorm1d(self.init_len // 8),
            nn.ReLU(True),

            nn.ConvTranspose1d(self.init_len // 8, self.init_len // 4, 1, bias=False),
            nn.BatchNorm1d(self.init_len // 4),
            nn.ReLU(True),

            nn.ConvTranspose1d(self.init_len // 4, self.init_len // 2, 1, bias=False),
            nn.BatchNorm1d(self.init_len // 2),
            nn.ReLU(True),

            nn.ConvTranspose1d(self.init_len // 2, self.init_len, 1, bias=False),
      )

    def forward(self, z):
        z = z.unsqueeze(-1)
        output = self.model(z)
        return output.squeeze(-1)

class PriorDiscriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.emb_dim = params.seq_len
        self.prior_dim = params.prior_dim
        self.l0 = nn.Linear(self.emb_dim, self.prior_dim)
        self.l1 = nn.Linear(self.prior_dim, self.prior_dim//4)
        self.l2 = nn.Linear(self.prior_dim//4, 1)
        
    def forward(self, x):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        return torch.sigmoid(self.l2(h))
