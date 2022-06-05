import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet
import cv2

from basnet_test import normPRED, save_output



#bounding box 찾기
#main
if __name__ == '__main__':
    #basnet test하기
    
    folder_list=os.listdir('###/101_ObjectCategories 데이터 위치###')
    os.mkdir('###/101_ObjectCategories 데이터 배경 제거 이미지 위치###/train/')
    os.mkdir('###/101_ObjectCategories 데이터 masking 위치###/train_pred/')
    #os.mkdir('/home/dircon/seoyun/test_result/train/new/')
    for i in folder_list:
        folder=i+'/'
        print(folder)
        #os.mkdir('/root/default/BASNet/test_result/val/'+folder)
        os.mkdir('###/101_ObjectCategories 데이터 배경 제거 이미지 위치###/train/'+folder)
        os.mkdir('###/101_ObjectCategories 데이터 masking 위치###/train_pred/'+folder)
        image_dir = '###/101_ObjectCategories 데이터 위치###/'+folder
        prediction_dir = '###/101_ObjectCategories 데이터 masking 위치###/train_pred/'+folder
        model_dir = '###basnet.pth 위치###'
        img_name_list = glob.glob(image_dir + '*.jpg')
        
        # --------- 2. dataloader ---------
        #1. dataload
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
        test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
        # --------- 3. model define ---------
        print("...load BASNet...")
        net = BASNet(3,1)
        net.load_state_dict(torch.load(model_dir))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):
            
            print("inferencing:",img_name_list[i_test].split("/")[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)
            
            d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            
            pred = normPRED(pred)
            
            # save results to test_results folder
            get=save_output(img_name_list[i_test],pred,prediction_dir)

            img=cv2.imread(img_name_list[i_test])
            mask=cv2.imread(get)
            masked=cv2.bitwise_and(img, mask)
            name=img_name_list[i_test].split("/")[-1]
            
            #name=name[0:len(name)-5]
            dir='###/101_ObjectCategories 데이터 위치###/train/'+folder+name
            cv2.imwrite(dir,masked)
    