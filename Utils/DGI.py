import numpy as np
from math import sqrt

def DGI(measurements,patterns,num_patterns,W,H):

    measurements = np.transpose(measurements)
    B_aver = 0
    SI_aver = 0
    R_aver = 0
    RI_aver = 0
    count = 0
    DGI = 0

    for i in range(num_patterns):

        pattern = patterns[i, :, :]
        count = count + 1
        B_r = measurements[i]

        SI_aver = (SI_aver * (count - 1) + pattern * B_r) / count  # 强度*
        B_aver = (B_aver * (count - 1) + B_r) / count  # 强度均值
        R_aver = (R_aver * (count - 1) + sum(sum(pattern))) / count
        RI_aver = (RI_aver * (count - 1) + sum(sum(pattern)) * pattern) / count
        DGI = SI_aver - B_aver / R_aver * RI_aver

    min_val = np.min(DGI)
    max_val = np.max(DGI)
    normed_data = (DGI - min_val) / (max_val - min_val)
    for i in range(W):
        for j in range(H):
            if normed_data[i][j]<0:
                normed_data[i][j]=0

    return normed_data,patterns












