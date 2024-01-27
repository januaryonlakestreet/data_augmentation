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
class mpii_pose_dataset(Dataset):
    def __init__(self,number_to_use="full"):

        self.scaler = MinMaxScaler()

        pose_json_data_location = "./dataset/mpii/mpii_annotations.json"
        self.pose_annotations = pd.read_json(pose_json_data_location)
        self.data = []
        self.img_data = []
        if number_to_use == "full":
            number_to_use = len(self.pose_annotations)

        for x in range(number_to_use):
            print(str(x) + "\\" + str(number_to_use))
            #self.data.append(torch.tensor(np.array(np.array(pose_annotations)[x][6])[:,:2].flatten()).to('cuda'))
            scaled_pose = self.scaler.fit_transform(np.array(np.array(self.pose_annotations)[x][6])[:,:2])

            img = Image.open(
                "./dataset/mpii/mpii_human_pose_v1/images/" + str(np.array(np.array(self.pose_annotations)[x][2])))

            img_size = img.size
            img_resize = img.resize((224, 224))
            pose_data = {
                "img_name":str(np.array(np.array(self.pose_annotations)[x][2])),
                "pose":torch.tensor(np.array(np.array(self.pose_annotations)[x][6])[:,:2]).flatten().to('cuda'),
                "pose_scaled":torch.tensor(np.array(scaled_pose).flatten()).to('cuda'),
                "img_loaded":np.array(img_resize).reshape((3,224,224)),
                "img_start_size": img_size
            }

            self.data.append(pose_data)

    def select_all_poses_by_img_name(self,img_name):
        return  self.pose_annotations[self.pose_annotations.iloc[:,2] == img_name]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class lsp_pose_dataset(Dataset):
    def __init__(self,number_to_use="full"):

        self.scaler = MinMaxScaler()

        pose_json_data_location = "./dataset/leeds-sport-pose-main/joints.mat"
        self.pose_annotations = loadmat(pose_json_data_location)
        self.data = []
        self.img_data = []
        if number_to_use == "full":
            number_to_use = len(self.pose_annotations)

        for x in range(1,number_to_use):
            print(str(x) + "\\" + str(number_to_use))
            formatted_number = '{:05d}'.format(x)

            img = Image.open("./dataset/leeds-sport-pose-main/images/im" + str(formatted_number)+'.jpg')
            pose = []
            for bone_idx in range(14):
                pose.append([self.pose_annotations['joints'][bone_idx][0][x-1],self.pose_annotations['joints'][bone_idx][1][x-1]])


            img_size = img.size
            img_resize = img.resize((224, 224))
            pose_data = {
                "img_name":str(formatted_number)+'.jpg',
                "pose":torch.tensor(np.array(pose).flatten()).to('cuda'),
                "img_loaded":np.array(img_resize).reshape((3,224,224)),
                "img_start_size": img_size
            }
            self.data.append(pose_data)






    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
