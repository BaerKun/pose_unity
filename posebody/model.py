import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from keys_map import keys_map


def load_dict_from_npy(weight_file):
    map_from_np2ts = keys_map

    try:
        np_weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        np_weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    ts_weights_dict = {}

    for k, v in np_weights_dict.items():
        ts_k = map_from_np2ts[k]
        if 'conv' in k:
            ts_weights_dict[ts_k + '.weight'] = torch.from_numpy(v['weights'])
            ts_weights_dict[ts_k + '.bias'] = torch.from_numpy(v['bias']).flatten()
        else:
            ts_weights_dict[ts_k + '.weight'] = torch.from_numpy(v['weights'])
    return ts_weights_dict


class Joint:
    def __init__(self, x=-1, y=-1, score: float = 0.):
        self.data = np.array((x, y, score), dtype=np.float32)

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def score(self):
        return self.data[2]

    @property
    def xy(self):
        return self.data[:2]

    def get_image_coord(self):
        return self.data[:2].astype(np.int32)


class Skeleton:
    def __init__(self, joints: list[Joint] = None, num_joints: int = 0, score: float = 0.):
        self.num_joints = num_joints
        self.score = score
        if joints is None:
            self.joints = [Joint()] * 25
        else:
            self.joints = joints

    def get_joints_coord(self) -> np.ndarray:
        return np.array([joint.xy for joint in self.joints])

    def resize(self, scale):
        for joint in self.joints:
            joint.data[:2] *= scale

    def __getitem__(self, item):
        return self.joints[item]

    def __setitem__(self, key, value: Joint):
        self.joints[key] = value


class SubBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(SubBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.prelu0 = nn.PReLU(num_parameters=growth_rate)
        self.conv1 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=growth_rate)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU(num_parameters=growth_rate)

    def forward(self, x):
        y1 = self.prelu0(self.conv0(x))
        y2 = self.prelu1(self.conv1(y1))
        y3 = self.prelu2(self.conv2(y2))
        return torch.cat((y1, y2, y3), dim=1)


def block(in_channels, growth_rate, hidden_channels, out_channels):
    midway_in_channels = 3 * growth_rate
    return nn.Sequential(OrderedDict(
        sub0=SubBlock(in_channels, growth_rate),
        sub1=SubBlock(midway_in_channels, growth_rate),
        sub2=SubBlock(midway_in_channels, growth_rate),
        sub3=SubBlock(midway_in_channels, growth_rate),
        sub4=SubBlock(midway_in_channels, growth_rate),
        conv0=nn.Conv2d(midway_in_channels, hidden_channels, 1),
        prelu0=nn.PReLU(num_parameters=hidden_channels),
        conv1=nn.Conv2d(hidden_channels, out_channels, 1)))


'''
x   ->  b00 -   -   -   -   -   -   -   -   -   -
        |       |       |       |       |       |
        |   ->  b10 ->  b11 ->  b12 ->  b13 -   -   -   -
                                        |       |       |
                                        b20 ->  b21 ->  y
'''


class PoseBody25(nn.Module):
    def __init__(self):
        super(PoseBody25, self).__init__()
        self.blk00 = nn.Sequential(OrderedDict(
            conv00=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), relu00=nn.ReLU(),
            conv01=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), relu01=nn.ReLU(),
            maxpool0=nn.MaxPool2d(kernel_size=2, stride=2),
            conv10=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), relu10=nn.ReLU(),
            conv11=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), relu11=nn.ReLU(),
            maxpool1=nn.MaxPool2d(kernel_size=2, stride=2),
            conv20=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), relu20=nn.ReLU(),
            conv21=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu21=nn.ReLU(),
            conv22=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu22=nn.ReLU(),
            conv23=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu23=nn.ReLU(),
            maxpool2=nn.MaxPool2d(kernel_size=2, stride=2),
            conv30=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), relu30=nn.ReLU(),
            conv31=nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            prelu31=nn.PReLU(num_parameters=512),
            conv32=nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            prelu32=nn.PReLU(num_parameters=256),
            conv33=nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            prelu33=nn.PReLU(num_parameters=128))
        )
        self.blk10 = block(128, 96, 256, 52)
        self.blk11 = block(180, 128, 512, 52)
        self.blk12 = block(180, 128, 512, 52)
        self.blk13 = block(180, 128, 512, 52)
        self.blk20 = block(180, 96, 256, 26)
        self.blk21 = block(206, 128, 512, 26)

    def forward(self, x):
        # layer 0
        y0 = self.blk00(x)

        # layer 1
        y1 = self.blk10(y0)
        x1 = torch.cat((y0, y1), 1)
        y1 = self.blk11(x1)
        x1 = torch.cat((y0, y1), 1)
        y1 = self.blk12(x1)
        x1 = torch.cat((y0, y1), 1)
        y1 = self.blk13(x1)

        # layer 2
        x2 = torch.cat((y0, y1), 1)
        y2 = self.blk20(x2)
        x2 = torch.cat((y0, y2, y1), 1)
        y2 = self.blk21(x2)

        return y2, y1
