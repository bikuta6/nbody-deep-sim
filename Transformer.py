import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from open3d.ml.torch.layers import ContinuousConv
import open3d.ml.torch as ml3d

class ContinuousConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContinuousConvEmbedding, self).__init__()
        self.conv = ContinuousConv(in_channels=in_channels, filters=out_channels, kernel_size=[3, 3, 3])

    def forward(self, features, pos_input, pos_output, extents):
        # Continuous convolution expects input shape [num_points, 3]
        x = self.conv(features, pos_input, pos_output, extents)

        return F.relu(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # Adding positional encoding to each timestep
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)

class ContConvTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, extents=3.0, calc_neighbors=True):
        super(ContConvTransformer, self).__init__()
        self.embedding = ContinuousConvEmbedding(in_channels=input_dim, out_channels=hidden_dim)
        self.pos_encoder = PositionalEncoding(embed_dim=hidden_dim)
        self.temporal_model = TransformerLayer(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.extents = extents
        self.calc_neighbors = calc_neighbors
        self.num_neighbors = None

    def forward(self, features):
        # Initialize a list to store the embeddings for each timestep
        all_embeddings = []
        
        # Process each timestep individually
        for t in range(len(features)):
            # Continuous convolution to get spatial embeddings at timestep t
            pos_input_t = features[t, :, :3]  # Extracting only the position coordinates
            spatial_embedding = self.embedding(
                features[t, :, :], pos_input_t, pos_input_t, self.extents)
            all_embeddings.append(spatial_embedding)

        if self.calc_neighbors:
            self.num_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv.nns.neighbors_row_splits)

        # Stack embeddings over the time dimension
        x = torch.stack(all_embeddings, dim=0)  # Shape: [timesteps, num_particles, hidden_dim]

        # Apply positional encoding for the temporal order
        x = self.pos_encoder(x)
        
        
        # Pass through the transformer layer
        x = self.temporal_model(x) # Shape: [timesteps, num_particles, hidden_dim]

        # Use only the last timestep's output for prediction
        x = self.fc(x[-1, :, :])  # Using the last timestep's output
        return x

'''
# Example usage
# Setting up sample input data for a single simulation
timesteps = 4  # Number of timesteps to process
num_particles = 502
input_dim = 4  # e.g.,velocity and mass
hidden_dim = 32
output_dim = 3  # e.g., acceleration prediction

# Features, positions, and extents for particles across timesteps
# Example dimensions: [timesteps, num_particles, input_dim]
features = torch.randn(timesteps, num_particles, input_dim)  # Particle features (e.g., position, velocity)
pos_input = torch.randn(timesteps, num_particles, 3)         # Input positions of particles
pos_output = pos_input.clone()                                 # Using the same for output positions
extents = 1.0         # Example extents (radius)

# Instantiate the model and make predictions
model = ParticlePredictionModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                    num_heads=4)

# Forward pass
import time
start = time.time()
for i in range(1000):
    output = model(features, pos_input, pos_output, extents, timesteps=timesteps)
end = time.time()
print("Time taken:", end - start)
print("Output shape:", output.shape)  # Expected shape: (num_particles, output_dim)
'''