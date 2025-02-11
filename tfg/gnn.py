import torch
from torch_geometric.nn import MLP
from torch_geometric.nn import GENConv, radius_graph, GravNetConv
from torch_geometric.data import Data, Batch
from torch.nn import Sequential, Linear, ReLU



def transform_to_graph(positions, features, y, radius=0.5, edge_attr=False, device='cuda', U=None, K=None):

    batch = []
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)

    for i in range(positions.size(0)):
        pos = positions[i]
        edge_index = radius_graph(pos, r=radius, batch=None, loop=False, max_num_neighbors=100)
        data = Data(pos=pos, edge_index=edge_index, x=torch.cat((pos, features[i]), dim=-1), y=y[i])
        batch_idx = torch.full((pos.size(0),), i, dtype=torch.long, device=device)
        data.batch = batch_idx
        if U is not None:
            data.U = U[i]
            data.K = K[i]

        if edge_attr:
            attr = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1).unsqueeze(-1)
            data.edge_attr = attr

        batch.append(data)

    return Batch.from_data_list(batch).to(device)


class GraphModel(torch.nn.Module):
    def __init__(self, input_dim=1, output_hiddens=None, output_dim=3, node_encoder_dims=None, edge_encoder_dims=None, gnn_dim=128, encoder_dropout=0.0, message_passing_steps=4, aggr='sum', device='cpu', G=1.0, softening=0.1, dt=0.01, radius=0.5):
        super(GraphModel, self).__init__()
        self.device = device
        self.G = G
        self.softening = softening
        self.dt = dt
        self.radius = radius
        self.node_encoder_dims = node_encoder_dims
        self.edge_encoder_dims = edge_encoder_dims

        if node_encoder_dims:
            self.node_encoder = MLP([input_dim] +  node_encoder_dims + [gnn_dim], norm=None, act='ReLU', device=device, dropout=encoder_dropout)
        else:
            self.node_encoder = torch.nn.Identity()

        if edge_encoder_dims:
            self.edge_encoder = MLP([1] + edge_encoder_dims, norm=None, act='ReLU', device=device, dropout=encoder_dropout)
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
                                        edge_dim=edge_encoder_dims[-1] if edge_encoder_dims else None))
            else:
                self.gnns.append(GENConv(in_channels=gnn_dim, 
                                        out_channels=gnn_dim, 
                                        aggr=aggr,
                                        num_layers=1,
                                        norm='layer',
                                        bias=True,
                                        eps=1e-7,
                                        edge_dim=edge_encoder_dims[-1] if edge_encoder_dims else None))
                
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
    
    def compute_neighbor_loss(self, acc, edges):
        """
        Computes the MSE loss between each node and the average predicted acceleration of its neighbors.
        
        Args:
            acc (Tensor): Predicted accelerations for the nodes, shape (N, 3), where N is the number of nodes.
            edges (Tensor): Edge index tensor, shape (2, E), where E is the number of edges. 
                            The first row contains the source nodes and the second row contains the destination nodes.

        Returns:
            Tensor: The MSE loss for the predicted accelerations between nodes and their neighbors.
        """
        # Initialize the tensor for the mean neighbor accelerations
        neighbor_acc_pred = torch.zeros_like(acc)

        # Loop over all nodes and calculate the mean acceleration of their neighbors
        for node_id in range(acc.size(0)):
            # Get neighbors for the current node (both directions: source -> destination and destination -> source)
            neighbors = torch.cat((edges[1, edges[0] == node_id], edges[0, edges[1] == node_id]))
            # remove duplicates
            neighbors = torch.unique(neighbors)
            
            # Avoid calculating mean if no neighbors (edge case)
            if neighbors.numel() > 0:
                neighbor_acc_pred[node_id] = acc[neighbors].mean(dim=0)  # Mean of neighbors' predicted accelerations
        
        # Compute MSE loss between node's predicted acceleration and its neighbors' mean predicted acceleration
        neighbor_loss = torch.nn.functional.mse_loss(acc, neighbor_acc_pred)
        
        return neighbor_loss
    
    def energy(self, positions, velocities, masses):
        x, y, z = positions[:, 0:1], positions[:, 1:2], positions[:, 2:3]
        dx = x.T - x  # Pairwise difference in x-coordinates
        dy = y.T - y  # Pairwise difference in y-coordinates
        dz = z.T - z  # Pairwise difference in z-coordinates

        # Compute pairwise squared distances with softening
        if self.softening == 0:
            r2 = dx**2 + dy**2 + dz**2
        else:
            r2 = dx**2 + dy**2 + dz**2 + self.softening**2  # Apply softening
        inv_r = torch.where(r2 > 0, r2**-0.5, torch.tensor(0.0, device=self.device))  # 1/r_ij

        # Compute potential energy (U) for pairwise interactions
        # We compute the potential energy between particles i and j
        U = -0.5 * self.G * torch.sum(masses[:, None] * masses * inv_r)  # Double sum over i, j

        # Compute kinetic energy (K)
        v2 = torch.sum(velocities**2, dim=1)  # Sum of squared velocities
        K = 0.5 * torch.sum(masses * v2)  # Kinetic energy

        return U, K
    
    def energy_loss(self, U, K, pred_acc, masses, initial_positions, initial_velocities):
        """
        Computes the energy loss between the predicted and stored energy values using leapfrog integration.
        
        Arguments:
        - U, K: Stored potential and kinetic energy from memory
        - pred_acc: Acceleration at the current timestep
        - masses: Particle masses
        - initial_positions: Positions from memory before integration step
        - initial_velocities: Velocities from memory before integration step
        
        Returns:
        - Energy loss based on MSE
        """

        # Half-step velocity update
        pred_velocities_half = initial_velocities + (pred_acc * self.dt / 2.0)

        # Full-step position update
        pred_positions = initial_positions + pred_velocities_half * self.dt

        # Compute new acceleration (should be done outside this function in a real simulation)
        # pred_acc_new = self.compute_acceleration(pred_positions, masses)  

        # Full-step velocity update
        pred_velocities = pred_velocities_half + (pred_acc * self.dt / 2.0)

        # Compute the predicted energy based on the new state
        pred_U, pred_K = self.energy(pred_positions, pred_velocities, masses)

        # Compute the energy loss
        energy_loss = torch.nn.functional.mse_loss(pred_U + pred_K, U + K)

        return energy_loss


    def compute_loss(self, data, acc_weigth=1.0, force_weigth=1.0, neighbor_weigth=1.0, energy_weigth=1.0):
        
        acc_pred = self.forward(data)

        num_graphs = data.num_graphs


        mse_losses = 0

        if neighbor_weigth == 0 and energy_weigth == 0:

            acc_loss = torch.nn.functional.mse_loss(acc_pred, data.y)
            mse_losses = acc_loss

            if force_weigth > 0:
                force_pred = data.x[:, 3].reshape(-1, 1) * acc_pred
                force_true = data.x[:, 3].reshape(-1, 1) * data.y
                force_loss = torch.nn.functional.mse_loss(force_pred, force_true)
                loss = acc_weigth * acc_loss + force_weigth * force_loss
            else:
                loss = acc_weigth * acc_loss

            return loss, mse_losses


        else:

            for i in range(num_graphs):
                idx = data.batch == i
                y = data.y[idx]
                acc = acc_pred[idx]
                feats = data.x[idx]
                pos = feats[:, :3]
                mass = feats[:, 3]
                vel = feats[:, 4:7]

                acc_loss = torch.nn.functional.mse_loss(acc, y)
                mse_losses += acc_loss
                loss = acc_weigth * acc_loss

                if force_weigth > 0:
                    # mass is of shape (N, 1) and acc is of shape (N, 3)
                    # F = ma
                    force_pred = mass.reshape(-1, 1) * acc
                    force_true = mass.reshape(-1, 1) * y
                    force_loss = torch.nn.functional.mse_loss(force_pred, force_true)
                    loss += force_weigth * force_loss

                if neighbor_weigth > 0:
                    # edge_indexes for this graph idx is the id of the nodes in the graph

                    edges = data.edge_index[:, data.batch[data.edge_index[0]] == i]

                    neighbor_loss = self.compute_neighbor_loss(acc, edges)
                    loss += neighbor_weigth * neighbor_loss

                if energy_weigth > 0:
                    energy_loss = self.energy_loss(data.U[i], data.K[i], acc, mass, pos, vel)
                    loss += energy_weigth * energy_loss

                if i == 0:
                    total_loss = loss
                else:
                    total_loss += loss

            return total_loss / num_graphs, mse_losses / num_graphs
    
    def train_batch(self, optimizer, pos, feat, acc, U=None, K=None, edge_attr=False, acc_weigth=1.0, force_weigth=1.0, neighbor_weigth=1.0, energy_weigth=1.0):
        self.train()
        optimizer.zero_grad()
        data = transform_to_graph(pos, feat, acc, edge_attr=edge_attr, device=self.device, U=U, K=K, radius=self.radius)
        loss, mse_loss = self.compute_loss(data, acc_weigth, force_weigth, neighbor_weigth, energy_weigth)
        loss.backward()
        optimizer.step()
        return loss.item(), mse_loss.item()
    
    def predict(self, pos, feat, edge_attr=False):
        self.eval()
        data = transform_to_graph(pos, feat, torch.zeros((pos.size(0), 3), device=self.device), edge_attr=edge_attr, device=self.device, radius=self.radius)
        return self.forward(data)
    



