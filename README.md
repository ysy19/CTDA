# CTDA
### 'Controlling Background Transparency for Data Augmentation' 구현 코드입니다.
## ○ Overview


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
