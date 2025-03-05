import torch
from torch_geometric.nn import radius, MLP
from torch_scatter import scatter
from torch import nn
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import time




class ContinuousConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_resolution=4, radius=0.5, agg='mean', self_loops=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.filter_resolution = filter_resolution
        self.filters = nn.Parameter(torch.randn(filter_resolution, filter_resolution, filter_resolution, in_channels, out_channels))
        self.agg = agg
        self.self_loops = self_loops
        self.neighbors = 0
        
    def ball_to_cube(self, r):
        norm = torch.norm(r, dim=-1, keepdim=True)
        r_unit = r / (norm + 1e-8)
        return r_unit * torch.tanh(norm)
    
    def trilinear_interpolate(self, coords):
        D = self.filter_resolution
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x0, y0, z0 = x.floor().long(), y.floor().long(), z.floor().long()
        x1, y1, z1 = (x0 + 1).clamp(max=D-1), (y0 + 1).clamp(max=D-1), (z0 + 1).clamp(max=D-1)
        xd, yd, zd = (x - x0.float()).view(-1, 1, 1), (y - y0.float()).view(-1, 1, 1), (z - z0.float()).view(-1, 1, 1)
        
        c000, c001 = self.filters[x0, y0, z0], self.filters[x0, y0, z1]
        c010, c011 = self.filters[x0, y1, z0], self.filters[x0, y1, z1]
        c100, c101 = self.filters[x1, y0, z0], self.filters[x1, y0, z1]
        c110, c111 = self.filters[x1, y1, z0], self.filters[x1, y1, z1]
        
        c00, c01, c10, c11 = c000 * (1 - zd) + c001 * zd, c010 * (1 - zd) + c011 * zd, c100 * (1 - zd) + c101 * zd, c110 * (1 - zd) + c111 * zd
        c0, c1 = c00 * (1 - yd) + c01 * yd, c10 * (1 - yd) + c11 * yd
        return c0 * (1 - xd) + c1 * xd
    
    def forward(self, positions, features, batch=None):
        batch = batch if batch else torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)
        
        edge_index = radius(positions, positions, self.radius, batch_x=batch, batch_y=batch, max_num_neighbors=10)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=positions.size(0))

        row, col = edge_index[0], edge_index[1]
        
        r = positions[col] - positions[row]
        dist2 = (r ** 2).sum(dim=-1)
        valid = (dist2 < self.radius**2).float()
        window = ((1 - dist2 / (self.radius**2)) ** 3) * valid
        
        mapped = self.ball_to_cube(r)
        grid_coords = (mapped + 1) * ((self.filter_resolution - 1) / 2)
        filt = self.trilinear_interpolate(grid_coords)
        
        conv_edge = torch.einsum('eio,ei->eo', filt, features[col])
        conv_edge = conv_edge * window.unsqueeze(1)
        
        output = scatter(conv_edge, row, dim=0, dim_size=positions.size(0), reduce=self.agg)
        return output
    


class ContinuousConvModel(nn.Module):

    def __init__(self, in_channels, out_channels, filter_resolution=[4], radius=0.5, agg='mean', self_loops=True, continuous_conv_layers=1,
                 continuous_conv_dim=64, continuous_conv_dropout=0.0,
                 encoder_hiddens=None, encoder_dropout=0.0, decoder_hiddens=None, decoder_dropout=0.0, device='cpu'):
        super(ContinuousConvModel, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_hiddens = encoder_hiddens
        self.encoder_dropout = encoder_dropout
        self.decoder_hiddens = decoder_hiddens
        self.decoder_dropout = decoder_dropout
        self.continuous_conv_layers = continuous_conv_layers
        self.continuous_conv_dim = continuous_conv_dim
        self.continuous_conv_dropout = continuous_conv_dropout
        self.neighbors = 0

        if encoder_hiddens:
            self.node_encoder = MLP([in_channels] + encoder_hiddens + [continuous_conv_dim], act='tanh', device=device, dropout=encoder_dropout)
        else:
            self.node_encoder = torch.nn.Identity()

        # Move encoder to device
        self.node_encoder = self.node_encoder.to(device)

        # Initialize GNN layers using ModuleList
        self.contconv = nn.ModuleList()
        for i in range(continuous_conv_layers):

            if type(filter_resolution) == list:
                filter_resolution_i = filter_resolution[i]
                if i == 0 and encoder_hiddens is None:
                    self.contconv.append(ContinuousConv(in_channels, continuous_conv_dim, filter_resolution_i, radius, agg, self_loops))
                else:
                    self.contconv.append(ContinuousConv(continuous_conv_dim, continuous_conv_dim, filter_resolution_i, radius, agg, self_loops))

            else:
                if i == 0 and encoder_hiddens is None:
                    self.gnns.append(ContinuousConv(in_channels, continuous_conv_dim, filter_resolution, radius, agg, self_loops))
                else:
                    self.gnns.append(ContinuousConv(continuous_conv_dim, continuous_conv_dim, filter_resolution, radius, agg, self_loops))

        self.contconv.to(device)

        if self.encoder_hiddens is None:
            out_dim = continuous_conv_dim + in_channels
        else:
            out_dim = continuous_conv_dim * 2

        self.layer_norm = nn.LayerNorm(out_dim).to(device)

        # Initialize output layers
        if decoder_hiddens:
            layers = []
            dims = [out_dim] + decoder_hiddens + [out_channels]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            self.output = nn.Sequential(*layers).to(device)
        else:
            self.output = nn.Linear(out_dim, out_channels).to(device)
        
    def forward(self, data):
        if self.in_channels == 4:
            x = torch.cat((data.x[:, :3], data.x[:, 6:]), dim=-1)
        else:
            x = data.x
        pos = x[:, :3]
        x = self.node_encoder(x)  # Always apply node encoding (Identity() if unused)

        encoder_output = x  # Store encoded node features

        for layer in self.contconv:
            x = layer(pos, x, data.batch)
            x = nn.Tanh()(x)
            x = nn.Dropout(self.continuous_conv_dropout)(x)

        # Concatenate encoded input and GNN output if needed
        x = torch.cat((encoder_output, x), dim=-1)

        x = self.layer_norm(x)

        return self.output(x)
    
    def predict(self, pos, x):
        data = Data(x=torch.cat([pos, x], dim=-1))
        return self.forward(data)
    
    def compute_loss(self, data):
        acc_pred = self.forward(data)
        loss = torch.nn.functional.mse_loss(acc_pred, data.y, reduction='mean')
        mse_losses = torch.nn.functional.mse_loss(acc_pred, data.y, reduction='mean')

        return loss, mse_losses
    
    def train_graph_batch(self, optimizer, data):
        self.train()
        optimizer.zero_grad()
        loss, mse_loss = self.compute_loss(data)
        loss.backward()
        optimizer.step()
        return loss.item(), mse_loss.item()
    
    def eval_graph_batch(self, data):
        self.eval()
        with torch.no_grad():
            start = time.time()
            acc_pred = self.forward(data)
            end = time.time()
            loss = torch.nn.functional.mse_loss(acc_pred, data.y, reduction='mean')
            mse_loss = torch.nn.functional.mse_loss(acc_pred, data.y, reduction='mean')
        return loss.item(), mse_loss.item(), end - start
    
