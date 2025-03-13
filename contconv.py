import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, MLP
from gnn import transform_to_graph
import time


class ContinuousConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, filter_resolution=4, radius=0.5, agg="mean"
    ):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.radius, self.agg = radius, agg
        self.filter_resolution = filter_resolution

        self.filters = nn.Parameter(
            torch.randn(
                filter_resolution,
                filter_resolution,
                filter_resolution,
                in_channels,
                out_channels,
            )
        )

    def ball_to_cube(self, r):
        norm = torch.norm(r, dim=-1, keepdim=True)
        r_unit = r / (norm + 1e-8)
        return r_unit * torch.tanh(norm)

    """
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
    """

    def trilinear_interpolate(self, coords):
        """
        Faster trilinear interpolation using grid_sample.

        coords: Tensor of shape (N, 3) with coordinates in [0, D-1].
        Returns: Tensor of shape (N, in_channels, out_channels)
        """
        D = self.filter_resolution
        # Normalize coords from [0, D-1] to [-1, 1]
        norm_coords = (coords / (D - 1)) * 2 - 1  # shape (N, 3)
        norm_coords = (
            norm_coords.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )  # (1, N, 1, 1, 3)

        # Reshape filters: from (D, D, D, in_channels, out_channels) to (1, in_channels*out_channels, D, D, D)
        filters_reshaped = (
            self.filters.view(D, D, D, -1).permute(3, 0, 1, 2).unsqueeze(0)
        )

        # Use grid_sample to interpolate: output shape (1, C, N, 1, 1) where C = in_channels*out_channels
        sampled = F.grid_sample(
            filters_reshaped, norm_coords, mode="bilinear", align_corners=True
        )
        sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)
        # Reshape back to (N, in_channels, out_channels)
        return sampled.view(-1, self.in_channels, self.out_channels)

    def forward(self, positions, features, edge_index):

        row, col = edge_index[0], edge_index[1]

        r = positions[col] - positions[row]
        dist2 = (r**2).sum(dim=-1)
        valid = (dist2 < self.radius**2).float()
        window = ((1 - dist2 / (self.radius**2)) ** 3) * valid

        mapped = self.ball_to_cube(r)
        grid_coords = (mapped + 1) * ((self.filter_resolution - 1) / 2)
        filt = self.trilinear_interpolate(grid_coords)
        conv_edge = torch.einsum("eio,ei->eo", filt, features[col])
        conv_edge = conv_edge * window.unsqueeze(1)

        output = scatter(
            conv_edge, row, dim=0, dim_size=positions.size(0), reduce=self.agg
        )
        return output


class ContinuousConvModel(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        filter_resolution=[4],
        radius=0.5,
        agg="mean",
        self_loops=True,
        continuous_conv_layers=1,
        continuous_conv_dim=64,
        continuous_conv_dropout=0.0,
        encoder_hiddens=None,
        encoder_dropout=0.0,
        decoder_hiddens=None,
        decoder_dropout=0.0,
        device="cuda",
        scale_factor=1,
    ):
        super().__init__()
        self.device, self.scale_factor = device, scale_factor
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
        self.radius = radius
        self.self_loops = self_loops

        if encoder_hiddens:
            self.node_encoder = MLP(
                [in_channels] + encoder_hiddens + [continuous_conv_dim],
                act="tanh",
                device=device,
                dropout=encoder_dropout,
            )
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
                    self.contconv.append(
                        ContinuousConv(
                            in_channels,
                            continuous_conv_dim,
                            filter_resolution_i,
                            self.radius,
                            agg,
                        )
                    )
                else:
                    self.contconv.append(
                        ContinuousConv(
                            continuous_conv_dim,
                            continuous_conv_dim,
                            filter_resolution_i,
                            self.radius,
                            agg,
                        )
                    )

            else:
                if i == 0 and encoder_hiddens is None:
                    self.gnns.append(
                        ContinuousConv(
                            in_channels,
                            continuous_conv_dim,
                            filter_resolution,
                            radius,
                            agg,
                        )
                    )
                else:
                    self.gnns.append(
                        ContinuousConv(
                            continuous_conv_dim,
                            continuous_conv_dim,
                            filter_resolution,
                            radius,
                            agg,
                        )
                    )

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
                layers.append(nn.Linear(dims[i], dims[i + 1]))
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
        batch = data.batch
        edge_index = radius_graph(pos, r=self.radius, batch=batch, loop=self.self_loops)
        x = self.node_encoder(x)
        encoder_output = x  # Store for concatenation
        for layer in self.contconv:
            x = layer(pos, x, edge_index)
            x = F.tanh(x)
            x = F.dropout(x, p=self.continuous_conv_dropout, training=self.training)

        x = self.layer_norm(torch.cat((encoder_output, x), dim=-1))
        return self.output(x)

    def compute_loss(self, data):
        acc_pred = self.forward(data)
        return torch.sqrt(
            F.mse_loss(acc_pred * self.scale_factor, data.y * self.scale_factor)
        ), F.mse_loss(acc_pred, data.y)

    def train_graph_batch(self, optimizer, data):
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
            loss = torch.sqrt(
                torch.nn.functional.mse_loss(acc_pred, data.y, reduction="mean")
            )
            mse_loss = torch.nn.functional.mse_loss(acc_pred, data.y, reduction="mean")
        return loss.item(), mse_loss.item(), end - start

    def predict(self, pos, feat):
        self.eval()
        with torch.no_grad():
            data = transform_to_graph(
                pos,
                feat,
                torch.zeros((pos.size(0), 3), device=self.device),
                device=self.device,
            )
            pred = self.forward(data)
        return pred
