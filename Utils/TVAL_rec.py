import os
import matlab
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import torch


def data_prepare(pattern_path,measurement_path,pattern_num,W,H,device):
    measurements = np.transpose(np.genfromtxt(measurement_path, delimiter=','))
    measurements = measurements / np.max(measurements)

    patterns_matrix = np.genfromtxt(pattern_path, delimiter=',')
    patterns_matrix = patterns_matrix[0:pattern_num, :]
    patterns = []
    for i in range(patterns_matrix.shape[0]):
        patterns.append(np.transpose(np.reshape(patterns_matrix[i, :], (W, H))))
    patterns = np.array(patterns)

    measurements = np.reshape(measurements, (1, pattern_num, 1, 1))
    patterns = torch.tensor(np.reshape(patterns, (pattern_num, 1, W, H))).float()

    return measurements,patterns

def list_files(directory):
    files_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            files_list.append(os.path.join(root, filename))
    return files_list

# Reconstruct target image with TVAL3.
def TVAL3(pattern_num,pattern_path,measurement_path):
    pwd_path = os.getcwd()
    files = list_files(os.path.join(pwd_path, 'TVAL_Utils'))
    eng = matlab.engine.start_matlab()
    eng.cd(pwd_path, nargout=0)
    for item in files:
        eng.addpath(item, nargout=0)
    print('MATLAB path modification complete.')
    image = np.array(eng.TVAL(pattern_num, pattern_path, measurement_path))
    eng.quit()
    return image