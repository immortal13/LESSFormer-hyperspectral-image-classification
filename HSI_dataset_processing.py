from osgeo import gdal_array
import scipy.io as sio
import numpy as np
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

## save xiongan_clip.mat
XA = 'xiongan.tif'
XA = gdal_array.LoadFile(XA)
XA = XA.transpose(1,2,0)[300:1100,1500:2500,:]
print(XA.shape) #(1580, 3750, 256) #(800, 1000, 256)

## save xiongan_clip_gt_new.mat
XA_gt = 'gt.tif'  
XA_gt = gdal_array.LoadFile(XA_gt)[300:1100,1500:2500]
print(XA_gt.shape) 

XA_gt[XA_gt==5] = 3
XA_gt[XA_gt==6] = 4
XA_gt[XA_gt==8] = 5
XA_gt[XA_gt==12] = 6
XA_gt[XA_gt==13] = 7
XA_gt[XA_gt==14] = 8
XA_gt[XA_gt==15] = 9
XA_gt[XA_gt==16] = 10
XA_gt[XA_gt==17] = 11
XA_gt[XA_gt==18] = 12
XA_gt[XA_gt==19] = 13
XA_gt[XA_gt==20] = 14

sio.savemat("xiongan_clip_gt_new.mat",{"xiongan_clip_gt":XA_gt})


## visualize flase-color image
# height, width, bands = XA.shape
# data = np.reshape(XA, [height * width, bands])
# minMax = preprocessing.StandardScaler()
# data = minMax.fit_transform(data)
# XA = np.reshape(data, [height, width, bands])

# b1 = XA[:,:,120]
# b2 = XA[:,:,72]
# b3 = XA[:,:,36]

# img = np.stack((b1, b2, b3),2)
# print(img.shape)
# plt.imshow(img)
# plt.show()


## count samples in each class
# print(np.unique(XA_gt))
# print(Counter(list(XA_gt.reshape(-1))))
# tmp = list(Counter(list(XA_gt.reshape(-1))).values())
# print(np.sum(tmp)-2247890) #total samples

"""
Counter({0: 2247890, 13: 1026513, 5: 475591, 4: 452144, 18: 421790, 1: 225647, 10: 193830, 2: 180766, 6: 169342, 
8: 165647, 15: 91072, 19: 65514, 12: 59165, 9: 38409, 20: 29616, 16: 29148, 7: 23304, 3: 15353, 14: 7151, 11: 5612, 17: 1496})
"""

"""
Counter({0: 267592, 13: 166164, 8: 70497, 6: 56363, 5: 52808, 
1: 49949, 15: 31485, 19: 29874, 12: 28761, 20: 14359, 2: 12816, 18: 10487, 16: 6046, 17: 1496, 14: 1303})
"""


