from Network.stage1_Unet import Stage1_UNet
import torch


# Learning rate decay strategy
def exponential_decay(initial_lr, global_step, decay_steps, decay_rate):
    return initial_lr * (decay_rate ** (global_step / decay_steps))

def total_variation(image):
    x_diff = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    y_diff = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return x_diff + y_diff

#Satge1 Loss
def Loss(out,groundtruth,image,TV_strength):
    loss1 = torch.nn.MSELoss()
    loss2 = total_variation(image)
    loss = loss1(out,groundtruth)+TV_strength*loss2
    return loss

class Stage1_model():
    def __init__(self, patterns,pattern_num,W,device,LR,TV_strength):
        super(Stage1_model, self).__init__()
        self.device = device
        self.LR = LR

        self.net = Stage1_UNet(patterns, pattern_num, W).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = Loss
        self.TV_strength = TV_strength

    def train_step(self,img,measurements,epoch):
        self.net.train()
        #data
        img = torch.tensor(img).float().to(self.device)
        measurements = torch.tensor(measurements, requires_grad=True).float().to(self.device)

        #Predication and Back propagation
        pred_image, pred_measurements = self.net(img)
        self.loss = self.loss_func(pred_measurements, measurements, pred_image, self.TV_strength)
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

        #learning rate update
        lr_temp = exponential_decay(self.LR, epoch, 100, 0.9)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr_temp

        return pred_image, pred_measurements, self.loss