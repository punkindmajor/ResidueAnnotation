import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        if torch.cuda.is_available():
            # Initialize CUDA device
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return x + self.encoding[:, :x.size(1)].detach().to(device)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        embedded = self.embedding(x)
        #embedded = embedded * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float))
        embedded = self.positional_encoding(embedded)
        embedded = embedded.permute(1, 0, 2)
        output = self.transformer(embedded)
        output = output.permute(1, 0, 2)
        output = self.fc(output)
        #output = self.pred_layer(output)
        return output