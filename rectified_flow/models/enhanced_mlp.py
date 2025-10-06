import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, t, time_factor=1000.0):
        t = t * time_factor  
        half_dim = self.num_channels // 2
        freqs = torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
        freqs = freqs / (half_dim - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        t = t.unsqueeze(1)  # (batch, 1)
        args = t * freqs.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (batch, num_channels)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, output_dim)
        )
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x, t_emb):
        h1 = self.act(self.linear1(x)) + self.time_mlp(t_emb)
        h2 = self.act(self.linear2(h1))
        return h2 + self.modify_x(x)


class MLPVelocity(nn.Module):
    def __init__(self, dim, 
                 encoder_dims=[512, 256, 128], 
                 decoder_dims=[128, 256, 512], 
                 time_emb_dim=128, 
                 output_dim=None):
        super().__init__()
        output_dim = output_dim or dim
        self.dim = dim
        self.time_emb_dim = time_emb_dim
        self.pos_emb = PositionalEmbedding(num_channels=time_emb_dim)
        self.act = nn.SiLU()
        
        self.input_layer = nn.Linear(dim + time_emb_dim, encoder_dims[0])
        
        self.encoder = nn.ModuleList()
        in_dim = encoder_dims[0]
        for h in encoder_dims[1:]:
            self.encoder.append(ResidualBlock(in_dim, h, t_dim=time_emb_dim))
            in_dim = h
        
        self.bottleneck = nn.Linear(in_dim, in_dim)
        
        self.decoder = nn.ModuleList()

        encoder_skips = [encoder_dims[0]] + encoder_dims[1:]
        encoder_skips.reverse()
        in_dim_decoder = in_dim
        for h in decoder_dims:
            skip_dim = encoder_skips.pop(0) if encoder_skips else 0
            self.decoder.append(ResidualBlock(in_dim_decoder + skip_dim, h, t_dim=time_emb_dim))
            in_dim_decoder = h
        
        self.output_layer = nn.Linear(in_dim_decoder, output_dim)

    def forward(self, x, t):
        t_emb = self.pos_emb(t)
        x = torch.cat([x, t_emb], dim=1)
        x = self.act(self.input_layer(x))
        skips = [x]
        for block in self.encoder:
            x = block(x, t_emb)
            skips.append(x)
        x = self.act(self.bottleneck(x))
        for block in self.decoder:
            if skips:
                skip = skips.pop() 
                x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
        x = self.output_layer(x)
        return x
