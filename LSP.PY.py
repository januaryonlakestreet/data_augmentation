import torch
from torch.utils.data import Dataset
from glob import glob
import scipy.io as sio
from PIL import Image
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
pose_json_data_location = "./dataset/leeds-sport-pose-main/labels.npz"
pose_json_data_locati1on = "./dataset/leeds-sport-pose-main/joints.mat"
a = np.load(pose_json_data_location)
b=  loadmat(pose_json_data_locati1on)


img = Image.open("./dataset/leeds-sport-pose-main/images/im00002.jpg")
plt.imshow(img)
for bone_idx in range(14):
    plt.plot(b['joints'][bone_idx][0][1], b['joints'][bone_idx][1][1], 'b+')
plt.show()
print("t")




