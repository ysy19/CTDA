# CTDA
### 'Controlling Background Transparency for Data Augmentation' 구현 코드입니다.
## ○ Overview
# Abstract
이미지 속의 객체는 대체로 특정한 배경 맥락 속에서 존재하기에, 네트워크가 학습할 때 객체가 가지고 있는 배경을 불가피하게 학습하게 된다. 하지만 학습에 긍정적 영향을 주는 배경도 있지만, 일반적인 배경이 아닐 경우 부정적인 영향을 주기도 한다. 따라서 이 연구에서는 이미지에서 배경의 투명도에 차이를 두어 배경의 영향력을 최적화할 수 있는지 검증해보고자 한다. Salient Object Detection 방법을 통해 배경과 객체를 분리하여 mask map을 생성하고 이를 토대로 투명도 10%~90%까지 설정한 이미지를 생성하였다. ImageNet과 Caltech101로 실험을 진행한 결과 비교적 Salient Object Detection 성능이 높은 Caltech101에서는 배경의 투명도를 조정한 이미지가 원본 이미지보다 더 높은 성능을 보이며, 배경의 영향력을 최적화할 수 있음을 보였다. 이를 통해 배경의 정보량을 조절할 때, 더 높은 성능을 가져올 수 있음을 확인하였다.

# Network
![image](https://user-images.githubusercontent.com/69154579/173506991-9f6bf821-9219-40c5-9132-c6097df96c24.png)

# Result 1) Images
![image](https://user-images.githubusercontent.com/69154579/173506882-2e6fce0c-99e6-4ff0-9016-5e7fe4c72759.png)
![image](https://user-images.githubusercontent.com/69154579/173506898-29047643-0139-4c81-8641-07fcc539709a.png)


# Result 2) Experiments
1. Caltech101 dataset 투명도 이미지 정확도

- ![image](https://user-images.githubusercontent.com/69154579/173506552-64565b9d-153d-4d91-b463-ffe6c92b2405.png)
ImageNet 투명도 조정 이미지 생성결과. 왼쪽 상단 투명도 10%부터 오른쪽 하단 투명도 90%.

- ![image](https://user-images.githubusercontent.com/69154579/173506609-bf70bb17-5d72-43ea-bf27-b1054633319f.png)
Caltech101 투명도 조정 이미지 생성결과. 왼쪽 상단 투명도 10%부터 오른쪽 하단 투명도 90%.

2. ImageNet dataset 투명도 이미지 정확도
![image](https://user-images.githubusercontent.com/69154579/173506657-84f6c668-01ab-48ac-8e74-093a1caf3898.png)
![image](https://user-images.githubusercontent.com/69154579/173506666-1415e609-2a0c-4036-ba81-054ec2f81db8.png)


## ○ Usage
1. Clone this repo
```
git clone https://github.com/ysy19/CTDA.
```
2. Download the pre-trained model basnet.pth from [GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) and put it into the dirctory 'saved_models/basnet_bsi/'

3. Download the data which you want : 'ImageNet' or 'Caltech101'

4. Edit the path of 'dataset' and 'the folder where you want to save mask map images' and 'the folder where you want to save masking images'  in the file.

5. Make and Train
5.1 Imagenet 
-  Cd to the directory 'BASNet'
-  Cd to the directory 'ImageNet_train'
-  Run the masking and making background deleted image process by command : 
```
python masking.py
```
-  Run training process by command : 
```
python train.py
```

5.1 Caltech101 
-  Cd to the directory 'BASNet'
-  Cd to the directory 'Caltech101_train'
-  Run the masking and making background deleted image process by command : 
```
python masking_101.py
```
-  Run training process by command : 
```
python train_101.py
```

## ○ References
- [1] Xuebin Qin, Deng-Ping Fan, Chenyang Huang, Cyril Diagne, Zichen Zhang, Adria Cabeza Sant’Anna, Albert Suarez, Martin Jagersand, and Ling Shao, “BASNet : Boundary-Aware Segmentation Network for Mobile and Web Applications”, In CVPR, 2019.
- BASNet 'https://github.com/xuebinqin/BASNet'

- [2] Jinwoo Choi, Chen Gao, Joseph C. E. Messou, Jia-Bin Huang, “Why Can’t I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition”, In NeurIPS, 2019.

- [3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li and Li Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database”, In CVPR, 2009.

- [4] Li Fei-Fei, Rob Fergus, Pietro Perona, “Learning Generative Visual Models from Few Training Examples : An Incremental Bayesian Approach Tested on 101 Object Categories.”, In CVPR, 2004.

- [5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, “Deep Residual Learning for Image Recognition”, In CVPR, 2016.
