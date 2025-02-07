import torch
from torch_geometric.nn import MLP
from torch_geometric.nn import GENConv, radius_graph
from torch_geometric.data import Data, Batch
from torch.nn import Sequential, Linear, ReLU



def transform_to_graph(positions, features, y, radius=0.5, edge_attr=False, device='cuda'):
    batch = []
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)

    for i in range(positions.size(0)):
        pos = positions[i]
        edge_index = radius_graph(pos, r=radius, batch=None, loop=False)
        data = Data(pos=pos, edge_index=edge_index, x=torch.cat((pos, features[i]), dim=-1), y=y[i])

        if edge_attr:
            attr = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1).unsqueeze(-1)
            data.edge_attr = attr

        batch.append(data)
        print(data)

    return Batch.from_data_list(batch).to(device)


class GraphModel(torch.nn.Module):
    def __init__(self, input_dim=1, output_hiddens=None, output_dim=3, node_encoder_dims=None, edge_encoder_dims=None, gnn_dim=256, encoder_dropout=0.0, message_passing_steps=4, aggr='sum', device='cpu'):
        super(GraphModel, self).__init__()
        self.device = device
        self.node_encoder_dims = node_encoder_dims
        self.edge_encoder_dims = edge_encoder_dims

        if node_encoder_dims:
            self.node_encoder = MLP([input_dim] +  node_encoder_dims + [gnn_dim], norm=None, act='ReLU', device=device, dropout=encoder_dropout)
        else:
            self.node_encoder = torch.nn.Identity()

        if edge_encoder_dims:
            self.edge_encoder = MLP([1] + edge_encoder_dims + [gnn_dim], norm=None, act='ReLU', device=device, dropout=encoder_dropout)
        else:
            self.edge_encoder = torch.nn.Identity()

        self.edge_encoder = self.edge_encoder.to(device)
        self.node_encoder = self.node_encoder.to(device)

        self.gnns = []
        for i in range(message_passing_steps):
            if i == 0 and node_encoder_dims is None:
                self.gnns.append(GENConv(in_channels=input_dim, 
                                        out_channels=gnn_dim, 
                                        aggr=aggr,
                                        num_layers=1,
                                        norm='layer',
                                        bias=True,
                                        eps=1e-7,
                                        edge_dim=gnn_dim if edge_encoder_dims else None))
            else:
                self.gnns.append(GENConv(in_channels=gnn_dim, 
                                        out_channels=gnn_dim, 
                                        aggr=aggr,
                                        num_layers=1,
                                        norm='layer',
                                        bias=True,
                                        eps=1e-7,
                                        edge_dim=gnn_dim if edge_encoder_dims else None))
                
        for gnn in self.gnns:
            gnn.to(device)
                
        if output_hiddens:
            layers = []
            dims = [gnn_dim] + output_hiddens + [output_dim]
            for i in range(len(dims) - 1):
                layers.append(Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(ReLU())
            self.output = Sequential(*layers).to(device)
        else:
            self.output = Linear(gnn_dim, output_dim).to(device)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        if self.edge_encoder_dims:
            edge_attr = self.edge_encoder(data.edge_attr)
        if self.node_encoder_dims:
            x = self.node_encoder(x)

        for gnn in self.gnns:
            if self.edge_encoder_dims:
                x = gnn(x, edge_index, edge_attr)
            else:
                x = gnn(x, edge_index)

        return self.output(x)
    