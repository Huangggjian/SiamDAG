# SiamDAG


Our work is based on the open source code library [TracKit](https://github.com/researchmm/TracKit), and the related operations of the environment configuration are exactly the same. Thanks to the contributors of Trackit.

## Contributors 
- [Jian Huang](https://github.com/Huangggjian)



## How To Start
- Tutorial for **SiamDAG**

  Follow SiamDAG **[[Training and Testing]](https://github.com/Huangggjian/SiamDAG/blob/main/lib/tutorial/DAG/dag.md)** tutorial 
- Our **SiamDAG** models and raw results

  Our models on VOT2019, OTB100, GOT10k can be downloaded [here:382E](https://pan.baidu.com/s/1L_gDJQQ1mVPZQAHXUYb2UA), and raw results can be downloaded [here:90v0](https://pan.baidu.com/s/16TKQnUnmT5jSOUkuDqIWew)

## Structure
- `experiments:` training and testing settings
- `dataset:` testing dataset
- `data:` training dataset
- `lib:` core scripts for all trackers
- `snapshot:` pre-trained models 
- `pretrain:` models trained on ImageNet (for training)
- `tutorials:` guidelines for training and testing
- `tracking:` training and testing interface

```
$TrackSeg
|—— experimnets
|—— lib
|—— snapshot
  |—— xxx.model/xxx.pth
|—— dataset
  |—— VOT2019.json 
  |—— VOT2019
     |—— ants1...
  |—— VOT2020
     |—— ants1...
|—— ...

```



## References
```
[1] Bhat G, Danelljan M, et al. Learning discriminative model prediction for tracking. ICCV2019.
[2] Chen, Kai and Wang, et.al. MMDetection: Open MMLab Detection Toolbox and Benchmark.
[3] Li, B., Wu, W., Wang, Q., et.al. Siamrpn++: Evolution of siamese visual tracking with very deep networks. CVPR2019.
[4] Dai, J., Qi, H., Xiong, Y., et.al. Deformable convolutional networks. ICCV2017.
[5] Wang, Q., Zhang, L., et.al. Fast online object tracking and segmentation: A unifying approach. CVPR2019.
[6] Vu, T., Jang, H., et.al. Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution. NIPS2019.
[7] VOT python toolkit: https://github.com/StrangerZhang/pysot-toolkit
```
## Contributors of TracKit
- **[Zhipeng Zhang](https://github.com/JudasDie)**
- **[Houwen Peng](https://houwenpeng.com/)**

