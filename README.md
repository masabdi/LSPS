Code for our BMVC oral paper: "3D Hand Pose Estimation using Simulation and Partial-Supervision with a Shared Latent Space"  see the paper at https://arxiv.org/abs/1807.05380


## Discriminative Results:
![Alt text](/img/dis_icvl.gif?raw=true "ICVL")

## Generative Results:
![Alt text](/img/walk_nyu.gif?raw=true "NYU")



#  Usage
1. Use pose_train to train the vae: 
```
python depth_train.py --config ../exps/nnyu.yaml
```

2. Pretrain the depth model using: 
```
python depth_train.py --config ../exps/nnyu.yaml --mode pretrain
```

3. Finally run this command for the unsupervised setting: 
```
python depth_train.py --config ../exps/nnyu.yaml --mode estimate3
```


