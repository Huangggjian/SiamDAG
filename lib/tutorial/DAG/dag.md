# DAG tutorial
## Testing

We assume the root path is $SiamDAG, e.g. `/home/hj/SiamDAG`
### Set up environment

```
cd $SiamDAG/lib/tutorial
bash install.sh $conda_path SiamDAG
cd $SiamDAG
conda activate SiamDAG

```
`$conda_path` denotes your anaconda path, e.g. `/home/hj/anaconda3`


### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/drive/folders/1DfiuFP2xuclVLzPkPKYkMWJXHKAZLJmk?usp=sharing)  to `$SiamDAG/snapshot`.
2. Download [json](https://drive.google.com/open?id=1S-RkzyMVRFWueWW91NmZldUJuDyhGdp1) files of testing data and put them in `$SiamDAG/dataset`.
3. Download testing data e.g. VOT2019 and put them in `$SiamDAG/dataset`. Please download each data from their official websites, and the directories should be named like `VOT2019`, `OTB2015`, `GOT10K`, `LASOT`.

### Testing
In root path `$SiamDAG`,

```
python tracking/test_DAG.py --arch DAG --resume snapshot/VOT2019.pth --dataset VOT2019 --align False --epoch_test True
```
### Evaluation
```
python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset VOT2019 --tracker_result_dir result/VOT2019 --trackers DAGVOT2019
```
You may test other datasets with our code. Please corresponds the provided pre-trained model `--resume` and dataset `--dataset`. 


### Reproduce ours results
Please download models from [here](https://pan.baidu.com/s/1L_gDJQQ1mVPZQAHXUYb2UA), code is 382E, and put them in $SiamDAG/snapshot, then do as Testing and Evaluation.


## Training
#### prepare data
- Please download training data from official websites,and you should refer to scripts in `$SiamDAG/lib/dataset/crop` to process your data. and then put them in `$SiamDAG/data`



#### prepare pretrained model
Please download the pretrained model on ImageNet [here](https://drive.google.com/open?id=1Pwe5NRdOoGiTYlnrOZdL-3S494RkbPQe), and then put it in `$SiamDAG/pretrain`.

#### modify settings
Please modify the training settings in `$SiamDAG/experiments/train/DAG.yaml`. The default number of GPU and batch size in paper are 3 and 32 respectively. 

#### run
In root path $SiamDAG,
```
python tracking/onekey.py
```
This script integrates **train**, **epoch test** and **tune**. It is suggested to run them one by one when you are not familiar with our whole framework (modify the key `ISTRUE` in `$SiamDAG/experiments/train/DAG.yaml`). When you know this framework well, simply run this one-key script.
or

```
python tracking/train_DAG.py --cfg experiments/train/DAG.yaml --gpus 0,1,2 --workers 16  
```
