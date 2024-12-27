'''
The code of Stage2_model is modified from https://github.com/nagejacob/SpatiallyAdaptiveSSID/blob/main/model/base.py
'''

from abc import abstractmethod
import os
import torch
from torch.nn.parallel import DataParallel

def build(obj_type, args):
    return obj_type(**args)

class BaseModel():
    def __init__(self, opt,device):
        self.opt = opt

        self.iter = 0 if 'iter' not in opt else opt['iter']

        self.networks = {}
        for network_opt in opt['networks']:
            Net = getattr(__import__('Network'), network_opt['type'])
            net = build(Net, network_opt['bss']).to(device)
            if 'path' in network_opt.keys():
                self.load_net(net, network_opt['path'])
            self.networks[network_opt['name']] = net

    @abstractmethod
    def train_step(self, data):
        pass

    @abstractmethod
    def validation_step(self, data):
        pass

    def data_parallel(self):
        for name in self.networks.keys():
            net = self.networks[name]
            net = net.cuda()
            net = DataParallel(net)
            self.networks[name] = net

    def save_net(self):
        for name, net in self.networks.items():
            if isinstance(net, DataParallel):
                net = net.module
            torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], '%s_iter_%08d.pth' % (name, self.iter)))

    def load_net(self, net, path):
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass
