import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c, ks=1):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_c, out_c, ks, bias=False))
        self.add_module("norm", nn.BatchNorm2d(out_c))
        self.add_module("RReLU", nn.RReLU())


class SimpleStack(nn.Module):
    def __init__(self):
        super(SimpleStack, self).__init__()
        c = [256, 128, 128]
        k = [1, 2, 2]
        views = [nn.Conv2d(2, c[0], k[0], bias=False)]
        for i in range(1, len(k)):
            views.append(nn.Conv2d(c[i - 1], c[i], k[i], bias=False))
        self.view = nn.Sequential(*views)
        c = [256, 256, 128, 128, 128, 128]
        k = [1, 3, 3, 3, 3, 3]
        maps = [nn.Conv2d(3, c[0], k[0], bias=False)]
        for i in range(1, len(k)):
            maps.append(nn.Conv2d(c[i - 1], c[i], k[i], bias=False))
        self.map = nn.Sequential(*maps)
        dm = [13953, 1024, 64, 3]
        decision_making = [nn.Linear(dm[0], dm[1]), nn.Sigmoid()]
        for i in range(1, len(dm) - 1):
            decision_making.append(nn.Linear(dm[i], dm[i + 1]))
            decision_making.append(nn.Sigmoid())
        self.decision_making = nn.Sequential(*decision_making)
        self._init_weight()

    def forward(self, income):
        view, whole_map, attitude = income
        view = self.view(view).flatten(start_dim=1)
        whole_map = self.map(whole_map).flatten(start_dim=1)
        # relative angle, distance to goal, distance sensor result
        all_features = [view, whole_map, attitude]
        all_features = torch.cat(all_features, dim=1)
        return self.decision_making(all_features)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
