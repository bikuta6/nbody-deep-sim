import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
import tqdm



    
class ContinuousConvolutionBlock(nn.Module):
    def __init__(self,
                 kernel_size=[4, 4, 4],
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 window_function=None,
                 in_channels=3,
                 out_channels=32,
                 filter_extent=0.1,
                 activation=None,
                 dropout=None,
                 calc_neighbors=False,
                 **kwargs):
        super(ContinuousConvolutionBlock, self).__init__()
        self.conv = ml3d.layers.ContinuousConv(
                            in_channels=in_channels,
                            filters=out_channels,
                            kernel_size=kernel_size,
                            align_corners=True,
                            interpolation=interpolation,
                            coordinate_mapping=coordinate_mapping,
                            normalize=False,
                            window_function=window_function,
                            radius_search_ignore_query_points=True,
                            **kwargs
                            )
        self.calc_neighbors = calc_neighbors
        self.extent = filter_extent
        self.dense = nn.Linear(in_channels, out_channels)
        self.num_neighbors = None

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        if activation:
            self.activation = F.relu

    def get_num_neighbors(self):
        '''
        This function is used to compute the number of neighbors for each particle.
        Only use after the forward pass.
        '''
        return ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv.nns.neighbors_row_splits)

    def forward(self, feats, pos):
        ans_conv = self.conv(feats, pos, pos, self.extent)
        if self.calc_neighbors:
            self.num_neighbors = self.get_num_neighbors()
        ans_dense = self.dense(feats)
        if hasattr(self, 'activation'):
            ans_conv = self.activation(ans_conv)
            ans_dense = self.activation(ans_dense)
        if hasattr(self, 'dropout'):
            ans_conv = self.dropout(ans_conv)
            ans_dense = self.dropout(ans_dense)

        return ans_conv, ans_dense




class Kaisarion(nn.Module):
    def __init__(
        self,
        kernel_size=[4, 4, 4],
        radius_scale=1,
        coordinate_mapping='ball_to_cube_volume_preserving',
        interpolation='linear',
        use_window=True,
        particle_radius=1,
        time_step=0.01,
        other_feats_channels=1 + 3 * 3, # mass + past pos * dim
        layer_channels=[32, 64, 64, 3],
        dropout=None,
        activation=None,
        calc_neighbors=True,
    ):

        super(Kaisarion, self).__init__()
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.time_step = time_step
        self.other_feats_channels = other_feats_channels
        self.layer_channels = layer_channels
        self.filter_extent = torch.tensor([self.radius_scale * 3 * self.particle_radius], dtype=torch.float32).item()
        self.dropout = dropout
        self.activation = activation
        self.calc_neighbors = calc_neighbors


        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)
        
        self.blocks = []


        self.convblock0 = ContinuousConvolutionBlock(kernel_size=self.kernel_size,
                                                coordinate_mapping=self.coordinate_mapping,
                                                interpolation=self.interpolation,
                                                window_function=window_poly6,
                                                in_channels=3 + self.other_feats_channels,
                                                out_channels=self.layer_channels[0],
                                                filter_extent=self.filter_extent,
                                                activation=self.activation,
                                                dropout=self.dropout,
                                                calc_neighbors=self.calc_neighbors)
        
        
        self.blocks = []
        self.blocks.append(self.convblock0)


        for i in range(1, len(self.layer_channels)-1):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch *= 2
            out_ch = self.layer_channels[i]
            block = ContinuousConvolutionBlock(kernel_size=self.kernel_size,
                                                coordinate_mapping=self.coordinate_mapping,
                                                interpolation=self.interpolation,
                                                window_function=window_poly6,
                                                activation=None,
                                                dropout=self.dropout,
                                                in_channels=in_ch,
                                                out_channels=out_ch,
                                                filter_extent=self.filter_extent)
            setattr(self, f'convblock{i}', block)
            self.blocks.append(block)
        
        self.convblock_last = ContinuousConvolutionBlock(kernel_size=self.kernel_size,
                                                coordinate_mapping=self.coordinate_mapping,
                                                interpolation=self.interpolation,
                                                window_function=window_poly6,
                                                in_channels=self.layer_channels[-2],
                                                out_channels=self.layer_channels[-1],
                                                filter_extent=self.filter_extent,
                                                activation=None)
        self.blocks.append(self.convblock_last)

        


    def forward(self, pos, vel, mass=None, extra_feats=None):
        if mass is not None:
            if mass.dim() == 1:
                mass = mass.unsqueeze(1)
            else:
                assert mass.dim() == 2


        feats = []
        if not mass is None:
            feats.append(mass)

        feats.append(vel)
        
        if not extra_feats is None:
            feats.append(extra_feats)

        feats = torch.cat(feats, axis=-1)

        self.ans_conv0, self.ans_dense0 = self.blocks[0](feats, pos)


        feats = torch.cat([
        self.ans_conv0, self.ans_dense0
        ],axis=-1)

        self.ans_convs = [feats]
        for block in self.blocks[1:]:
            inp_feats = self.ans_convs[-1]
            ans_conv, ans_dense = block(inp_feats, pos)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            ans = F.relu(ans)
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_neighbors = self.convblock0.num_neighbors

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.accelerations = self.ans_convs[-1]
        return self.accelerations





