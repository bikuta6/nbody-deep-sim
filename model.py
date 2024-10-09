import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
import tqdm

class KwisatzHaderach(nn.Module):
    def __init__(
        self,
        kernel_size=[4, 4, 4],
        radius_scale=1.5,
        coordinate_mapping='ball_to_cube_volume_preserving',
        interpolation='linear',
        use_window=True,
        particle_radius=0.1,
        time_step=0.01,
        other_feats_channels=1, # mass
        layer_channels=[32, 64, 64, 3]
    ):

        super(KwisatzHaderach, self).__init__()
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

        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = ml3d.layers.ContinuousConv(
                           kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs
                           )

            self._all_convs.append((name, conv))
            return conv

        self.conv0 = Conv("conv0", in_channels=3 + self.other_feats_channels, filters=self.layer_channels[0])
        self.dense0 = nn.Linear(in_features=3 + self.other_feats_channels, out_features=self.layer_channels[0])

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch *= 2
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)

    def get_displacements(self, pos, vel, mass=None):
        if mass is not None:
            if mass.dim() == 1:
                mass = mass.unsqueeze(1)
            else:
                assert mass.dim() == 2
    
        # compute the extent of the filters (the diameter)
        filter_extent = self.filter_extent

        feats = [vel]
        if not mass is None:
            feats.append(mass)
        feats = torch.cat(feats, axis=-1)

        self.ans_conv0= self.conv0(feats, pos, pos,
                                                filter_extent)
        self.ans_dense0 = self.dense0(feats)


        feats = torch.cat([
        self.ans_conv0, self.ans_dense0
        ],axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_displacement = self.ans_convs[-1]
        return self.pos_displacement

    def forward(self, pos, vel, mass=None):
        displacement = self.get_displacements(pos, vel, mass)

        new_pos = pos + displacement
        new_vel = displacement / self.time_step

        return new_pos, new_vel

'''
import time

dummy_pos = torch.rand(1000, 3)
dummy_vel = torch.rand(1000, 3)
dummy_mass = torch.rand(1000, 1)
print(dummy_pos.shape, dummy_vel.shape, dummy_mass.shape, dummy_mass.dim())

model = KwisatzHaderach()
start = time.time()
new_pos, new_vel = model(dummy_pos, dummy_vel, dummy_mass)
for i in tqdm.tqdm(range(1000)):
    new_pos, new_vel = model(new_pos, new_vel, dummy_mass)

end = time.time()

print("Time taken for 1000 iterations: ", end - start)

print(new_pos.shape, new_vel.shape)
'''