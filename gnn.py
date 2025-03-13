# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import MLP
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, Tanh, ModuleList, LayerNorm
import time


def transform_to_graph(positions, features, y, batch=None, neighbors=50, device="cuda"):

    graph = knn_graph(positions, k=neighbors, batch=batch, loop=False)
    data = Data(
        x=torch.cat((positions, features), dim=-1),
        edge_index=graph,
        edge_attr=None,
        y=y,
        batch=batch,
    )

    return data


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_hiddens=None,
        output_dim=3,
        node_encoder_dims=None,
        gnn_dim=128,
        encoder_dropout=0.0,
        message_passing_steps=4,
        aggr="sum",
        device="cpu",
        neighbors=50,
        scale_factor=1,
    ):
        super(GraphModel, self).__init__()

        # Store parameters
        self.device = device
        self.neighbors = neighbors
        self.node_encoder_dims = node_encoder_dims
        self.message_passing_steps = message_passing_steps
        self.aggr = aggr
        self.output_hiddens = output_hiddens
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.gnn_dim = gnn_dim
        self.encoder_dropout = encoder_dropout
        self.scale_factor = scale_factor

        # Initialize node encoder
        if node_encoder_dims:
            self.node_encoder = MLP(
                [input_dim] + node_encoder_dims + [gnn_dim],
                act="tanh",
                device=device,
                dropout=encoder_dropout,
                norm=None,
            )
        else:
            self.node_encoder = torch.nn.Identity()

        # Move encoder to device
        self.node_encoder = self.node_encoder.to(device)

        # Initialize GNN layers using ModuleList
        self.gnns = ModuleList()
        for i in range(message_passing_steps):
            if i == 0 and node_encoder_dims is None:
                self.gnns.append(
                    EdgeConv(
                        nn=Sequential(
                            Linear(input_dim * 2, gnn_dim),
                            Tanh(),
                            Linear(gnn_dim, gnn_dim),
                        ),
                        aggr=aggr,
                    )
                )
            else:
                self.gnns.append(
                    EdgeConv(
                        nn=Sequential(
                            Linear(gnn_dim * 2, gnn_dim),
                            Tanh(),
                            Linear(gnn_dim, gnn_dim),
                        ),
                        aggr=aggr,
                    )
                )
        self.gnns.to(device)

        if self.node_encoder_dims is None:
            out_dim = gnn_dim + input_dim
        else:
            out_dim = gnn_dim * 2

        self.layer_norm = LayerNorm(out_dim).to(device)

        # Initialize output layers
        if output_hiddens:
            layers = []
            dims = [out_dim] + output_hiddens + [output_dim]
            for i in range(len(dims) - 1):
                layers.append(Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(Tanh())
            self.output = Sequential(*layers).to(device)
        else:
            self.output = Linear(out_dim, output_dim).to(device)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_hiddens": self.output_hiddens,
            "output_dim": self.output_dim,
            "node_encoder_dims": self.node_encoder_dims,
            "gnn_dim": self.gnn_dim,
            "encoder_dropout": self.encoder_dropout,
            "message_passing_steps": self.message_passing_steps,
            "aggr": self.aggr,
            "device": self.device,
            "neighbors": self.neighbors,
        }

    def forward(self, data):
        if self.input_dim == 4:
            x = torch.cat((data.x[:, :3], data.x[:, 6:]), dim=-1)
        else:
            x = data.x
        x = self.node_encoder(x)  # Always apply node encoding (Identity() if unused)

        edge_index = data.edge_index
        encoder_output = x  # Store encoded node features

        for gnn in self.gnns:
            x = gnn(x, edge_index)

        # Concatenate encoded input and GNN output if needed
        x = torch.cat((encoder_output, x), dim=-1)

        x = self.layer_norm(x)

        return self.output(x)

    def compute_loss(self, data):
        acc_pred = self.forward(data)
        loss = torch.sqrt(
            torch.nn.functional.mse_loss(
                acc_pred * self.scale_factor,
                data.y * self.scale_factor,
                reduction="mean",
            )
        )
        mse_losses = torch.nn.functional.mse_loss(acc_pred, data.y, reduction="mean")

        return loss, mse_losses

    def train_batch(self, optimizer, pos, feat, acc):
        self.train()
        optimizer.zero_grad()

        batch = torch.cat(
            [
                torch.full((pos[i].size(0),), i, dtype=torch.long, device=self.device)
                for i in range(len(pos))
            ]
        )

        # pos is (batch_size, n_bodies, 3), eliminate the batch dimension
        pos = pos.reshape(-1, 3)
        feat = feat.reshape(-1, feat.size(-1))
        acc = acc.reshape(-1, 3)

        data = transform_to_graph(pos, feat, acc, batch=batch, device=self.device)
        loss, mse_loss = self.compute_loss(data)
        loss.backward()
        optimizer.step()
        return loss.item(), mse_loss.item()

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

    def predict_graph(self, data):
        self.eval()
        with torch.no_grad():
            pred = self.forward(data)
        return pred

    def step(self, pos, vel, m, acc, dt):
        # Calculamos las velocidades en medio paso...
        vel_ = vel + 0.5 * dt * acc
        # ...luego las posiciones con esas velocidades...
        pos_ = pos + dt * vel_
        # ...luego las nuevas aceleraciones en las posiciones actualizadas...
        acc_ = self.model.predict(pos_, torch.cat([vel_, m], dim=-1))
        # ...y por último actualizamos de nuevo las posiciones
        vel_ += 0.5 * dt * acc_
        return pos_, vel_, acc_

    def rollout(self, pos, vel, m, steps, dt):
        self.eval()

        if m.dim() == 1:
            m = m.unsqueeze(-1)

        pos_ = pos
        vel_ = vel
        memory = {"pos": [pos_], "vel": [vel_]}

        with torch.no_grad():
            acc = self.predict(pos_, torch.cat((pos_, vel_, m), dim=-1))
            memory["acc"] = [acc]
            for _ in range(steps):
                pos_, vel_, acc = self.step(pos_, vel_, m, acc, dt)
                memory["pos"].append(pos_)
                memory["vel"].append(vel)
                memory["acc"].append(acc)

        return memory
