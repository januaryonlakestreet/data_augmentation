import random

from dataset import mpii_pose_dataset,lsp_pose_dataset
from torch.utils.data import DataLoader
from setting import *
from networks import Encoder,Decoder,Pose_Discriminator

import numpy as np


import matplotlib.pyplot as plt
from PIL import Image


import torch
import torch.nn as nn
import torchvision.models as models
def calculate_percentage_correct(detected_keypoints, ground_truth_keypoints):
    correct_count = 0
    head_correct = 0
    shoulder_correct = 0
    elbow_correct = 0
    wrist_correct = 0
    hip_correct = 0
    knee_correct = 0
    ankle_correct = 0
    # lsp
    # Right ankle 0
    # Right knee 1
    # Right hip 2
    # Left hip 3
    # Left knee 4
    # Left ankle
    # Right wrist
    # Right elbow 7
    # Right shoulder 8
    # Left shoulder 9
    # Left elbow 10
    # Left wrist 11
    # Neck
    # Head top'''
    batch_size = len(ground_truth_keypoints)

    for batch_idx in range(batch_size):
        gt = np.array(ground_truth_keypoints.detach().cpu())[batch_idx]
        gt = gt.reshape((14,2))
        tolerance =calculate_distance(gt[2],gt[3]) * 0.2

        dt = np.array(detected_keypoints.detach().cpu())[batch_idx]
        dt = dt.reshape((14,2))


        for i, (detected_keypoint, ground_truth_keypoint) in enumerate(zip(dt, gt)):
            distance = calculate_distance(detected_keypoint, ground_truth_keypoint)
            if distance <= tolerance:
                if i in [12,13]:
                    head_correct += 1
                if i in [8,9]:
                    shoulder_correct += 1
                if i in [7, 10]:
                    elbow_correct += 1
                if i in [6, 11]:
                    wrist_correct += 1
                if i in [2, 3]:
                    hip_correct += 1
                if i in [1, 4]:
                    knee_correct += 1
                if i in [0, 5]:
                    ankle_correct += 1

    max_keypoints = 2 * batch_size

    head_correctp = (head_correct / max_keypoints) * 100
    shoulder_correctp = (shoulder_correct / max_keypoints) * 100
    elbow_correctp = (elbow_correct / max_keypoints) * 100
    wrist_correctp = (wrist_correct / max_keypoints) * 100
    hip_correctp = (hip_correct / max_keypoints) * 100
    knee_correctp = (knee_correct / max_keypoints) * 100
    ankle_correctp = (ankle_correct / max_keypoints) * 100

    return head_correctp, shoulder_correctp,elbow_correctp,wrist_correctp,hip_correctp,knee_correctp,ankle_correctp

def calculate_distance(point1, point2):
    # Calculate the distance between two keypoints (e.g., Euclidean distance)
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def mpjpe(gt,pred):
    gt_ = []
    pred_ = []
    for _ in range(BATCH_SIZE):
        __ = np.array(gt[_].detach().cpu()).reshape((14,2))
        gt_.append(__)
    for _ in range(BATCH_SIZE):
        __ = np.array(pred[_].detach().cpu()).reshape((14, 2))
        pred_.append(__)
    gt_ = np.array(gt_)
    pred_ = np.array(pred_)

    # Calculate Euclidean distance between predicted and target poses for each joint
    joint_errors = np.linalg.norm(pred_ - gt_, axis=-1)

    # Calculate MPJPE for each sample
    mpjpe_per_sample = np.mean(joint_errors, axis=1)

    # Calculate overall MPJPE across all samples and joints
    mpjpe = np.mean(mpjpe_per_sample)

    return mpjpe
class PoseDetectionModel(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseDetectionModel, self).__init__()
        self.num_keypoints = num_keypoints
        # Load a pre-trained model, for example, resnet-18
        self.backbone = models.resnet18(weights="default")
        # Replace the classification layer to match the number of keypoints
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_keypoints * 2)

        # Add Batch Normalization after the backbone
        self.batch_norm = nn.BatchNorm1d(self.num_keypoints * 2)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)  # Adjust dropout probability as needed

    def forward(self, x):
        x = self.backbone(x)
        return x


class AttentionExpert(nn.Module):
    def __init__(self, num_classes=64, num_experts=3):
        super(AttentionExpert, self).__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.experts = nn.ModuleList([models.resnet18() for _ in range(self.num_experts)])
        for expert in self.experts:
            expert.fc = nn.Linear(expert.fc.in_features, 28)

    def forward(self, x):
        experts_outputs = [expert(x) for expert in self.experts]

        legs = experts_outputs[0][:,:12]
        body = experts_outputs[1][:,24:28]
        arms = experts_outputs[2][:,12:24]

        return torch.cat((legs,body,arms),dim=1)

def plot_mpii(outputs,labels,epoch,data):
    plt.clf()
    outputs_formatted = []
    labels_formatted = []
    for batch_idx in range(BATCH_SIZE):
        outputs_formatted.append(np.array(outputs[batch_idx].detach().cpu()).reshape((14, 2)))
        labels_formatted.append(np.array(labels[batch_idx].detach().cpu()).reshape((14, 2)))

    plt.plot(0, 0, 'b+', label="predicted key points")
    plt.plot(0, 0, 'r+', label="ground truth key points")
    # r ankle_X,r ankle_Y,
    # r knee_X,r knee_Y,
    # r hip_X,r hip_Y,
    # l hip_X,l hip_Y,
    # l knee_X,l knee_Y,
    # l ankle_X,l ankle_Y,
    # pelvis_X,pelvis_Y,
    # thorax_X,thorax_Y,
    # upper neck_X,upper neck_Y,
    # head top_X,head top_Y,
    # r wrist_X,r wrist_Y,
    # r elbow_X,r elbow_Y,
    # r shoulder_X,r shoulder_Y,
    # l shoulder_X,l shoulder_Y,
    # l elbow_X,l elbow_Y,
    # l wrist_X,l wrist_Y

    for bone_idx in range(14):
        plt.plot(outputs_formatted[0][bone_idx][0], outputs_formatted[0][bone_idx][1], 'b+')
    for bone_idx in range(len(labels_formatted)):
        plt.plot(labels_formatted[0][bone_idx][0], labels_formatted[0][bone_idx][1], 'r+')
    plt.legend()
    img = Image.open("./dataset/leeds-sport-pose-main/images/im" + data['img_name'][0])
    plt.imshow(img)
    plt.savefig('./generated_images/' + str(epoch) + "_" + str(random.randint(0, 1000)) + '.png')
def plot_lsp(outputs,labels,epoch,data):
    plt.clf()
    outputs_formatted = []
    labels_formatted = []
    for batch_idx in range(BATCH_SIZE):
        outputs_formatted.append(np.array(outputs[batch_idx].detach().cpu()).reshape((14, 2)))
        labels_formatted.append(np.array(labels[batch_idx].detach().cpu()).reshape((14, 2)))

    tmp_xr = [outputs_formatted[0][0][0], outputs_formatted[0][1][0]]
    tmp_yr = [outputs_formatted[0][0][1], outputs_formatted[0][1][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][1][0], outputs_formatted[0][2][0]]
    tmp_yr = [outputs_formatted[0][1][1], outputs_formatted[0][2][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][2][0], outputs_formatted[0][3][0]]
    tmp_yr = [outputs_formatted[0][2][1], outputs_formatted[0][3][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][5][0], outputs_formatted[0][4][0]]
    tmp_yr = [outputs_formatted[0][5][1], outputs_formatted[0][4][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][3][0], outputs_formatted[0][4][0]]
    tmp_yr = [outputs_formatted[0][3][1], outputs_formatted[0][4][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][8][0], outputs_formatted[0][2][0]]
    tmp_yr = [outputs_formatted[0][8][1], outputs_formatted[0][2][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][9][0], outputs_formatted[0][3][0]]
    tmp_yr = [outputs_formatted[0][9][1], outputs_formatted[0][3][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][9][0], outputs_formatted[0][8][0]]
    tmp_yr = [outputs_formatted[0][9][1], outputs_formatted[0][8][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')

    tmp_xr = [outputs_formatted[0][10][0], outputs_formatted[0][9][0]]
    tmp_yr = [outputs_formatted[0][10][1], outputs_formatted[0][9][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][10][0], outputs_formatted[0][11][0]]
    tmp_yr = [outputs_formatted[0][10][1], outputs_formatted[0][11][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')

    tmp_xr = [outputs_formatted[0][7][0], outputs_formatted[0][8][0]]
    tmp_yr = [outputs_formatted[0][7][1], outputs_formatted[0][8][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')
    tmp_xr = [outputs_formatted[0][6][0], outputs_formatted[0][7][0]]
    tmp_yr = [outputs_formatted[0][6][1], outputs_formatted[0][7][1]]
    plt.plot(tmp_xr, tmp_yr, 'b')


    #real pose
    tmp_xr = [labels_formatted[0][0][0], labels_formatted[0][1][0]]
    tmp_yr = [labels_formatted[0][0][1], labels_formatted[0][1][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][1][0], labels_formatted[0][2][0]]
    tmp_yr = [labels_formatted[0][1][1], labels_formatted[0][2][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][2][0], labels_formatted[0][3][0]]
    tmp_yr = [labels_formatted[0][2][1], labels_formatted[0][3][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][5][0], labels_formatted[0][4][0]]
    tmp_yr = [labels_formatted[0][5][1], labels_formatted[0][4][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][3][0], labels_formatted[0][4][0]]
    tmp_yr = [labels_formatted[0][3][1], labels_formatted[0][4][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][8][0], labels_formatted[0][2][0]]
    tmp_yr = [labels_formatted[0][8][1], labels_formatted[0][2][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][9][0], labels_formatted[0][3][0]]
    tmp_yr = [labels_formatted[0][9][1], labels_formatted[0][3][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][9][0], labels_formatted[0][8][0]]
    tmp_yr = [labels_formatted[0][9][1], labels_formatted[0][8][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][10][0], labels_formatted[0][9][0]]
    tmp_yr = [labels_formatted[0][10][1], labels_formatted[0][9][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][10][0], labels_formatted[0][11][0]]
    tmp_yr = [labels_formatted[0][10][1], labels_formatted[0][11][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')

    tmp_xr = [labels_formatted[0][7][0], labels_formatted[0][8][0]]
    tmp_yr = [labels_formatted[0][7][1], labels_formatted[0][8][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')
    tmp_xr = [labels_formatted[0][6][0], labels_formatted[0][7][0]]
    tmp_yr = [labels_formatted[0][6][1], labels_formatted[0][7][1]]
    plt.plot(tmp_xr, tmp_yr, 'r')


    plt.plot(0, 0, 'b+', label="predicted key points")
    plt.plot(0, 0, 'r+', label="ground truth key points")


    for bone_idx in range(14):
        plt.plot(outputs_formatted[0][bone_idx][0], outputs_formatted[0][bone_idx][1], 'b+')
    for bone_idx in range(len(labels_formatted)):
        plt.plot(labels_formatted[0][bone_idx][0], labels_formatted[0][bone_idx][1], 'r+')
    plt.legend()
    img = Image.open("./dataset/leeds-sport-pose-main/images/im" + data['img_name'][0])
    plt.imshow(img)
    plt.savefig('./generated_images/' + str(epoch) + "_" + str(random.randint(0, 1000)) + '.png')

def train_model(model,discriminator, dataloader, optimizer, criterion, num_epochs=EPOCH):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_mpjpe = 0.0
        running_pck = []
        for step,data in enumerate(dataloader):
            inputs = data['img_loaded'].float().to('cuda')
            labels = data['pose'].float().to('cuda')

            optimizer_discriminator.zero_grad()
            for _ in range(5):
                truth_labels = torch.ones(BATCH_SIZE).float().to('cuda')
                discriminator_on_real_poses = discriminator(labels)
                loss_real = criterion(discriminator_on_real_poses,truth_labels)
                loss_real.backward()

                fake_poses = model(inputs)
                fake_labels = torch.zeros(BATCH_SIZE).float().to('cuda')
                discriminator_on_fake_poses = discriminator(fake_poses).squeeze()
                loss_fake = criterion(discriminator_on_fake_poses,fake_labels)
                loss_fake.backward()

                discriminator_loss = loss_real + loss_fake
                optimizer_discriminator.step()


            #'generator'
            optimizer.zero_grad()
            outputs = model(inputs)
            output_discriminator = discriminator(outputs).squeeze()
            loss = criterion(output_discriminator, truth_labels) + 0.2 * criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mpjpe += mpjpe(outputs,labels)
            running_pck.append(calculate_percentage_correct(outputs,labels))

        epoch_loss = running_loss / len(dataloader)
        epoch_mpjpe = running_mpjpe / len(dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}  discriminator loss: {discriminator_loss.item()}  mpjpe: {epoch_mpjpe:.4f} "
              f"PoKs head@2.0: {np.array(running_pck).mean(axis=0)[0]:.4f} "
              f"PoKs shoulders: {np.array(running_pck).mean(axis=0)[1]:.4f} "
              f"PoKs elbow: {np.array(running_pck).mean(axis=0)[2]:.4f} "
              f"PoKs wrist: {np.array(running_pck).mean(axis=0)[3]:.4f} "
              f"PoKs hips: {np.array(running_pck).mean(axis=0)[4]:.4f} "
              f"PoKs knees: {np.array(running_pck).mean(axis=0)[5]:.4f} "
              f"PoKs ankles: {np.array(running_pck).mean(axis=0)[6]:.4f} "
              f"PoKs mean: {np.array(running_pck).mean()}"
              )


        if epoch % 1000 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'saved_models/resnet_'+str(epoch)+'.pth')
            torch.save(discriminator.state_dict(), 'saved_models/discriminator_' + str(epoch) + '.pth')

        show = True
        if show:
            if epoch % 100 == 0 and epoch != 0:
                plot_lsp(outputs,labels,epoch,data)



if __name__ == '__main__':
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))
    else:
        print("CUDA NOT RUNNING!")
    device = torch.device("cuda:0" if CUDA else "cpu")

    dataset = lsp_pose_dataset(100)
    data_loader = DataLoader(dataset.data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

    num_keypoints = 14 # use 16 for mpii
    #model = PoseDetectionModel(num_keypoints).to('cuda')
    model = AttentionExpert(num_keypoints).to('cuda')
    discriminator = Pose_Discriminator().to('cuda')

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LR)
    criterion = nn.MSELoss()  # You might need to adjust the loss function based on your task

    load_model = True
    if load_model:
        model.load_state_dict(torch.load('saved_models/lsp/resnet_10000.pth'))
        discriminator.load_state_dict(torch.load('saved_models/lsp/discriminator_10000.pth'))
    train_model(model,discriminator, data_loader, optimizer, criterion, num_epochs=EPOCH)
