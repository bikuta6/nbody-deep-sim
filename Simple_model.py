import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d


class ContinuousConvolutionModel(nn.Module):  # Inherit from nn.Module

    def __init__(self,
                 conv_filters=[64, 64, 32],
                 fc_sizes=[64, 64, 32],
                 use_dense_in_conv=False,
                 activation=F.relu,
                 kernel_size=[6, 6, 6],
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 window_function=None,
                 num_features=4,
                 output_dim=3,
                 radius=3.0,
                 calc_neighbors=False):
        super(ContinuousConvolutionModel, self).__init__()
        self.activation = activation
        self.num_features = num_features
        self.radius = radius
        self.use_dense_in_conv = use_dense_in_conv
        self.conv_filters = conv_filters
        self.fc_sizes = fc_sizes
        self.kernel_size = kernel_size
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.window_function = window_function
        self.output_dim = output_dim
        self.calc_neighbors = calc_neighbors

        # Use nn.ModuleList to register layers
        self.convs = nn.ModuleList()
        for i in range(len(conv_filters)):
            in_channels = num_features if i == 0 else conv_filters[i-1]
            out_channels = conv_filters[i]
            self.convs.append(ml3d.layers.ContinuousConv(
                kernel_size=kernel_size,
                coordinate_mapping=coordinate_mapping,
                interpolation=interpolation,
                window_function=window_function,
                in_channels=in_channels,
                filters=out_channels,
                activation=activation,
                dense=use_dense_in_conv
            ))

        self.fc = nn.ModuleList()
        for i in range(len(fc_sizes)):
            in_channels = conv_filters[-1] if i == 0 else fc_sizes[i-1]
            out_channels = fc_sizes[i]
            self.fc.append(nn.Linear(in_channels, out_channels))

        self.out_layer = nn.Linear(fc_sizes[-1], output_dim)

    def forward(self, feats, pos):
        for conv in self.convs:
            feats = conv(inp_features=feats, inp_positions=pos, out_positions=pos, extents=self.radius)

        if self.calc_neighbors:
            self.num_neighbors = ml3d.ops.reduce_subarrays_sum(
                torch.ones_like(self.convs[0].nns.neighbors_index,
                                dtype=torch.float32),
                self.convs[0].nns.neighbors_row_splits)

        for fc in self.fc:
            feats = fc(feats)
            feats = self.activation(feats)

        feats = self.out_layer(feats)
            
        return feats

