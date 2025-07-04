import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm,trange
from PIL import Image
from Utils.TVAL_rec import TVAL3,data_prepare
from model.Stage1_model import Stage1_model
from model.stage2_model import Stage2_model
import os
from Utils.parse import parse



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pattern_num = int(args.SR * args.W * args.H)

    # Prepocess intensity and patterns: normalize intensity and reshape patterns with shape of (pattern_num, width, height), ToTensor
    measurements, patterns = data_prepare(args.pattern_path, args.measurement_path, pattern_num, args.W, args.H, device)


    '''
    Reconstruct target image with TVAL3 (Two methods):
        1. Directly reconstruct the image by calling the TVAL3 (MATLAB version) through Python.
        2. In MATLAB, reconstruct the images using TVAL3 and save them. Read the reconstructed images as input for the network. 
        
        If any errors occur while configuring the Python-MATLAB connection in the first method, choose the second method.
        The code of TVAL3 (MATLAB version) is from: https://github.com/larzw/TVAL3
    '''

    '''Method1: Directly reconstruct the image by calling the TVAL3 (MATLAB version).'''
    image = TVAL3(pattern_num, args.pattern_path, args.measurement_path)
    plt.imsave(os.path.join(args.save_path,'TVAL3_rec.bmp'), image, cmap='gray')
    image = np.reshape(image,(1, 1, args.W,args.H))

    '''Method2: Read the reconstructed image as input.'''
    # image = np.array(Image.open('data/rec/TVAL3_rec.bmp').convert('L'))
    # image = np.reshape(image,(1, 1, args.W, args.H))


    #Stage1: Physical model-driven image reconstruction.
    stage1_model = Stage1_model(patterns.to(device),pattern_num,args.W,device,args.LR[0],args.TV_strength)


    for epoch in tqdm(range(args.EPOCH[0]),desc='Satge1'):
        pred_image, pred_measurements, loss=stage1_model.train_step(image,measurements,epoch)
        if epoch % 10 == 0:
            print('loss:{}'.format(loss))
            plt.subplot(221)
            plt.imshow(np.squeeze(image))
            plt.title('TVAL')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(222)
            plt.imshow(np.reshape(pred_image.cpu().detach().numpy(), (args.W, args.H)))
            plt.title('Physical model-driven')
            plt.xticks([])
            plt.yticks([])

            ax1 = plt.subplot(223)
            plt.plot(np.reshape(pred_measurements.cpu().detach().numpy(), (pattern_num)))
            plt.title('pred_y')
            ax1.set_aspect(1.0 / ax1.get_data_ratio())

            ax2 = plt.subplot(224)
            plt.plot(np.reshape(measurements, (pattern_num)))
            plt.title('GT_y')
            ax2.set_aspect(1.0 / ax2.get_data_ratio())
            plt.show()
        plt.imsave(os.path.join(args.save_path, 'Stadge1_rec.bmp'),
                   np.reshape(pred_image.cpu().detach().numpy(), (args.W, args.H)), cmap='gray')



    # Stage2: Self-supervised denoising.
    opt = args.opt
    Model = getattr(__import__('model'), opt['model'])
    stage2_model = Model(opt,device,args.LR[1],args.EPOCH)

    noisy = pred_image.detach().clone()

    for epoch in tqdm(range(sum(args.EPOCH[1:])),desc='Satge2'):
        stage, loss, rec = stage2_model.train_step(noisy)
        if epoch % 40 == 0:
            print("Current subnetwork: {}  loss: {}".format(stage,loss.cpu().detach().numpy()))
            plt.subplot(121)
            plt.imshow(np.reshape(noisy.cpu().detach().numpy(), (args.W, args.H)))
            plt.title('Noise Image')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(122)
            plt.imshow(np.reshape(rec.cpu().detach().numpy(), (args.W, args.H)))
            plt.title('Denoising Image')
            plt.xticks([])
            plt.yticks([])
            plt.show()


    plt.imsave(os.path.join(args.save_path, 'Stadge2_denoising.bmp'),
               np.reshape(rec.cpu().detach().numpy(), (args.W, args.H)), cmap='gray')





def parse_args():
    parser = argparse.ArgumentParser(description="PESS training")
    parser.add_argument("--measurement_path", default=r"data/Intensity/example.csv", type=str,help="The path of 1D intensity.")
    parser.add_argument("--pattern_path", default=r"data/pattern/m_128_0.08.csv", type=str,help="The path of pattern.")
    parser.add_argument("--save_path", default=r"data/rec", type=str, help="The save path of reconstructed images.")
    parser.add_argument("--EPOCH", default=[80,61,100,100], type=int, help="The total epoch of training.")
    parser.add_argument("--LR", default=[0.005,0.001], type=float, help="Learning rate.")
    parser.add_argument("--TV_strength", default=1e-8, type=float, help="Learning rate.")
    parser.add_argument("--SR", default=0.08008, type=float, help="Sampling rate.")
    parser.add_argument("--W", default=128, type=int, help="Width.")
    parser.add_argument("--H", default=128, type=int, help="Height.")
    parser.add_argument('--opt',default= parse('model/stage2.json'),help='The size of blindspot.')
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    main(args)

