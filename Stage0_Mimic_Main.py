# 模型训练尝试
import argparse
import torch
import datetime
import json
import yaml
import os

# from Data_Preparation.data_preparation import Data_Preparation
from Stage0_Mimic_DDPM import DDPM
from Stage0_Mimic_UnetCF import MultiUNet
from Stage0_Mimic_TrainCF import train
from Stage0_Mimic_EvaluateCF import evaluate
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# 数据、标签原始路径
datafolder = 'H:/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
patientfolder = 'H:/Paper03_Data/PTBXL_100Hz/'
foldername = './ddpmModel/Stage0_MimicBig_NEW'
path_Ori = datafolder
path_pat = patientfolder


# 参数设置
Batch_Size = 128
Epochs = 100
Feats = 80
dim = 64
Learning_Rate = 1.0e-3
Beta_Start = 0.0001
Beta_End = 0.5
Num_Steps = 50
Schedule = "quad"
with open("config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

# 检测CUDA是否可用，如果可用，使用第二块GPU，否则使用CPU
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = True
# 数据读取

route01 = 'H:/Paper03_Data/MIMIC4withLabel/big_embedding.npy'
# route01 = 'H:/Paper03_Data/MIMIC4withLabel/val_embedding.npy'
desc_Train_Ori = np.load(route01, allow_pickle=True)
# route1 = 'H:/Paper03_Data/New_100Hz/New_100_Train_Data.npy'
route1 = 'H:/Paper03_Data/MIMIC4withLabel/big_data.npy'
# route1 = 'H:/Paper03_Data/MIMIC4withLabel/val_data.npy'
X_Train_Ori = np.load(route1, allow_pickle=True)
# X_Train_Data = X_Train_Ori[:, :, :]
# Y_Train_Data = X_Train_Ori[:, :, 1:2]
X_Train_Data = X_Train_Ori[:, :, :]
Y_Train_Data = X_Train_Ori[:, :, 0:1]
# Y_Train_Data = np.tile(Y_Train_Data, (1, 1, 12))
X_Train_Data = np.transpose(X_Train_Data, (0, 2, 1))
Y_Train_Data = np.transpose(Y_Train_Data, (0, 2, 1))


# route1 = 'H:/Paper03_Data/New_100Hz/New_100_Train_Data.npy'
# X_Train_Ori = np.load(route1, allow_pickle=True)
# X_Train_Data = X_Train_Ori[:, :, :]
# Y_Train_Data = X_Train_Ori[:, :, 1:2]
# # X_Train_Data = X_Train_Ori[:, :, :]
# # Y_Train_Data = X_Train_Ori[:, 1:2, :]
# # Y_Train_Data = np.tile(Y_Train_Data, (1, 1, 12))
# X_Train_Data = np.transpose(X_Train_Data, (0, 2, 1))
# Y_Train_Data = np.transpose(Y_Train_Data, (0, 2, 1))

route02 = 'H:/Paper03_Data/MIMIC4withLabel/val_embedding.npy'
desc_Val_Ori = np.load(route02, allow_pickle=True)
route2 = 'H:/Paper03_Data/MIMIC4withLabel/val_data.npy'
X_Val_Ori = np.load(route2, allow_pickle=True)
X_Val_Data = X_Val_Ori[:, :, :]
Y_Val_Data = X_Val_Ori[:, :, 0:1]
# Y_Val_Data = np.tile(Y_Val_Data, (1, 1, 12))
X_Val_Data = np.transpose(X_Val_Data, (0, 2, 1))
Y_Val_Data = np.transpose(Y_Val_Data, (0, 2, 1))

route03 = 'H:/Paper03_Data/MIMIC4withLabel/test_embedding.npy'
desc_Test_Ori = np.load(route03, allow_pickle=True)
route3 = 'H:/Paper03_Data/MIMIC4withLabel/test_data.npy'
X_Test_Ori = np.load(route3, allow_pickle=True)
X_Test_Data = X_Test_Ori[:, :, :]
Y_Test_Data = X_Test_Ori[:, :, 0:1]
# Y_Test_Data = np.tile(Y_Test_Data, (1, 1, 12))
X_Test_Data = np.transpose(X_Test_Data, (0, 2, 1))
Y_Test_Data = np.transpose(Y_Test_Data, (0, 2, 1))

X_Train_Data = torch.FloatTensor(X_Train_Data)
Y_Train_Data = torch.FloatTensor(Y_Train_Data)
D_Train_Data = torch.FloatTensor(desc_Train_Ori)
X_Val_Data = torch.FloatTensor(X_Val_Data)
Y_Val_Data = torch.FloatTensor(Y_Val_Data)
D_Val_Data = torch.FloatTensor(desc_Val_Ori)
X_Test_Data = torch.FloatTensor(X_Test_Data)
Y_Test_Data = torch.FloatTensor(Y_Test_Data)
D_Test_Data = torch.FloatTensor(desc_Test_Ori)

del X_Train_Ori, X_Val_Ori, X_Test_Ori, desc_Train_Ori, desc_Val_Ori, desc_Test_Ori

train_set = TensorDataset(X_Train_Data, Y_Train_Data, D_Train_Data)
val_set = TensorDataset(X_Val_Data, Y_Val_Data, D_Val_Data)
test_set = TensorDataset(X_Test_Data, Y_Test_Data, D_Test_Data)

train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=50, num_workers=0)

# print("！分隔符01！")

# base_model = ConditionalModel(Feats).to(DEVICE)
# base_model = UNet().to(DEVICE)
base_model = MultiUNet().to(DEVICE)
model = DDPM(base_model, config, DEVICE)
train(model, config['train'], train_loader, DEVICE,
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)
# print("！分隔符02！")

# print('eval final')
# evaluate(model, val_loader, 1, DEVICE, foldername=foldername)
#
# # eval best
print('eval best')
output_path = foldername + "/model_9.pth"
model.load_state_dict(torch.load(output_path))
evaluate(model, test_loader, 1, DEVICE, foldername=foldername)

print('eval best')
output_path = foldername + "/model_16.pth"
model.load_state_dict(torch.load(output_path))
evaluate(model, test_loader, 1, DEVICE, foldername=foldername)

print('eval best')
output_path = foldername + "/model_30.pth"
model.load_state_dict(torch.load(output_path))
evaluate(model, test_loader, 1, DEVICE, foldername=foldername)

print('eval best')
output_path = foldername + "/model_42.pth"
model.load_state_dict(torch.load(output_path))
evaluate(model, test_loader, 1, DEVICE, foldername=foldername)

print('eval best')
output_path = foldername + "/model_88.pth"
model.load_state_dict(torch.load(output_path))
evaluate(model, test_loader, 1, DEVICE, foldername=foldername)

# don't use before final model is determined
print('eval final')
output_path = foldername + "/final.pth"
model.load_state_dict(torch.load(output_path))
evaluate(model, test_loader, 1, DEVICE, foldername=foldername)
