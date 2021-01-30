import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0,1,2"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

config.DAG = edict()
config.DAG.TRAIN = edict()
config.DAG.TEST = edict()
config.DAG.TUNE = edict()
config.DAG.DATASET = edict()
config.DAG.DATASET.VID = edict()
config.DAG.DATASET.GOT10K = edict()
config.DAG.DATASET.COCO = edict()
config.DAG.DATASET.DET = edict()
config.DAG.DATASET.LASOT = edict()
config.DAG.DATASET.YTB = edict()
config.DAG.DATASET.VISDRONE = edict()

# augmentation
config.DAG.DATASET.SHIFT = 4
config.DAG.DATASET.SCALE = 0.05
config.DAG.DATASET.COLOR = 1
config.DAG.DATASET.FLIP = 0
config.DAG.DATASET.BLUR = 0
config.DAG.DATASET.GRAY = 0
config.DAG.DATASET.MIXUP = 0
config.DAG.DATASET.CUTOUT = 0
config.DAG.DATASET.CHANNEL6 = 0
config.DAG.DATASET.LABELSMOOTH = 0
config.DAG.DATASET.ROTATION = 0
config.DAG.DATASET.SHIFTs = 64
config.DAG.DATASET.SCALEs = 0.18
config.DAG.DATASET.NEG = 0.2

# vid
config.DAG.DATASET.VID.PATH = '$data_path/vid/crop511'
config.DAG.DATASET.VID.ANNOTATION = '$data_path/vid/train.json'

# got10k
config.DAG.DATASET.GOT10K.PATH = '$data_path/got10k/crop511'
config.DAG.DATASET.GOT10K.ANNOTATION = '$data_path/got10k/train.json'
config.DAG.DATASET.GOT10K.RANGE = 100
config.DAG.DATASET.GOT10K.USE = 200000

# visdrone
config.DAG.DATASET.VISDRONE.ANNOTATION = '$data_path/visdrone/train.json'
config.DAG.DATASET.VISDRONE.PATH = '$data_path/visdrone/crop271'
config.DAG.DATASET.VISDRONE.RANGE = 100
config.DAG.DATASET.VISDRONE.USE = 100000

# train
config.DAG.TRAIN.GROUP = "resrchvc"
config.DAG.TRAIN.EXID = "setting1"
config.DAG.TRAIN.MODEL = "DAG"
config.DAG.TRAIN.RESUME = False
config.DAG.TRAIN.START_EPOCH = 0
config.DAG.TRAIN.END_EPOCH = 50
config.DAG.TRAIN.TEMPLATE_SIZE = 127
config.DAG.TRAIN.SEARCH_SIZE = 255
config.DAG.TRAIN.STRIDE = 8
config.DAG.TRAIN.BATCH = 32
config.DAG.TRAIN.PRETRAIN = 'pretrain.model'
config.DAG.TRAIN.LR_POLICY = 'log'
config.DAG.TRAIN.LR = 0.001
config.DAG.TRAIN.LR_END = 0.00001
config.DAG.TRAIN.MOMENTUM = 0.9
config.DAG.TRAIN.WEIGHT_DECAY = 0.0001
config.DAG.TRAIN.WHICH_USE = ['GOT10K']  # VID or 'GOT10K'
config.DAG.TRAIN.NEG_NUM = 16
config.DAG.TRAIN.POS_NUM = 16
config.DAG.TRAIN.TOTAL_NUM = 64
# test
config.DAG.TEST.MODEL = config.DAG.TRAIN.MODEL
config.DAG.TEST.DATA = 'VOT2019'
config.DAG.TEST.START_EPOCH = 30
config.DAG.TEST.END_EPOCH = 50

# tune
config.DAG.TUNE.MODEL = config.DAG.TRAIN.MODEL
config.DAG.TUNE.DATA = 'VOT2019'
config.DAG.TUNE.METHOD = 'TPE'  # 'GENE' or 'RAY'



def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB', 'LASOT']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    try:
                        config[model_name][k][vk][vvk] = vvv
                    except:
                        config[model_name][k][vk] = edict()
                        config[model_name][k][vk][vvk] = vvv

    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['DAG', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=DAG or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
