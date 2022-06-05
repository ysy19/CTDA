# CTDA
'Controlling Background Transparency for Data Augmentation' 구현 코드입니다.

## Usage
### 1. Clone this repo
```
git clone https://github.com/ysy19/CTDA
```
### 2. Download the pre-trained model basnet.pth from [GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu)

### 3. Download the data which you want : 'ImageNet' or 'Caltech101'

### 4.
#### 4.1 Imagenet 
-  Cd to the directory 'BASNet'
-  Cd to the directory 'ImageNet_train'
-  Run the masking and making background deleted image process by command : 
'''
python masking.py
'''
-  Run training process by command : 
'''
python train.py
'''

#### 4.1 Caltech101 
-  Cd to the directory 'BASNet'
-  Cd to the directory 'Caltech101_train'
-  Run the masking and making background deleted image process by command : '''python masking_101.py'''
-  Run training process by command : '''python train_101.py

## ○ 참고문서
- BASNet 'https://github.com/xuebinqin/BASNet'
