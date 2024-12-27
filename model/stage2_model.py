'''
The code of Stage2_model is modified from https://github.com/nagejacob/SpatiallyAdaptiveSSID/blob/main/network/bnn.py
'''
from model.base import BaseModel
import os
import torch
import torch.nn as nn


def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    return img

def generate_alpha(input, lower=1, upper=5):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()

    return ratio

class Stage2_model(BaseModel):
    def __init__(self, opt,device,LR,EPOCH):
        super(Stage2_model, self).__init__(opt,device)
        self.stage = None
        self.iter = 0
        self.MS_BNN_iters = EPOCH[1]
        self.Trans_LAN_iters = EPOCH[2]
        self.UNet_iters = EPOCH[3]

        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_MS_BNN = torch.optim.Adam(self.networks['MS_BNN'].parameters(), lr=LR)
        self.scheduler_MS_BNN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_MS_BNN, self.MS_BNN_iters)
        self.optimizer_Trans_LAN = torch.optim.Adam(self.networks['Trans_LAN'].parameters(), lr=LR)
        self.scheduler_Trans_LAN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_Trans_LAN, self.Trans_LAN_iters)
        self.optimizer_UNet = torch.optim.Adam(self.networks['UNet'].parameters(), lr=LR)
        self.scheduler_UNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_UNet, self.UNet_iters)

    def train_step(self, data):
        self.iter += 1
        self.update_stage()
        input = data

        if self.stage == 'MS_BNN':
            self.networks['MS_BNN'].train()
            x = self.networks['MS_BNN'](input)
            self.loss = self.criteron(x, input)

            self.optimizer_MS_BNN.zero_grad()
            self.loss.backward()
            self.optimizer_MS_BNN.step()
            self.scheduler_MS_BNN.step()
            self.out = x

        elif self.stage == 'Trans_LAN':
            self.networks['MS_BNN'].eval()
            self.networks['Trans_LAN'].train()
            with torch.no_grad():
                MS_BNN = self.networks['MS_BNN'](input)
            Trans_LAN = self.networks['Trans_LAN'](input)

            alpha = generate_alpha(MS_BNN)
            self.loss = self.criteron(MS_BNN.detach() * (1 - alpha), Trans_LAN * (1 - alpha))
            # self.loss = self.criteron(MS_BNN, Trans_LAN)
            self.optimizer_Trans_LAN.zero_grad()
            self.loss.backward()
            self.optimizer_Trans_LAN.step()
            self.scheduler_Trans_LAN.step()
            self.out = Trans_LAN

        elif self.stage == 'UNet':
            self.networks['MS_BNN'].eval()
            self.networks['Trans_LAN'].eval()
            self.networks['UNet'].train()
            with torch.no_grad():
                MS_BNN = self.networks['MS_BNN'](input)
                Trans_LAN = self.networks['Trans_LAN'](input)
            UNet = self.networks['UNet'](input)

            alpha = generate_alpha(MS_BNN)
            self.loss = self.criteron(MS_BNN * (1 - alpha), UNet * (1 - alpha)) + self.criteron(Trans_LAN * alpha, UNet * alpha)
            self.optimizer_UNet.zero_grad()
            self.loss.backward()
            self.optimizer_UNet.step()
            self.scheduler_UNet.step()
            self.out = UNet
        return self.stage,self.loss,self.out

    def update_stage(self):
        if self.iter <= self.MS_BNN_iters:
            self.stage = 'MS_BNN'
        elif self.iter <= self.MS_BNN_iters + self.Trans_LAN_iters:
            self.stage = 'Trans_LAN'
        else:
            self.stage = 'UNet'
