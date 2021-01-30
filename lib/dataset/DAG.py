# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by HUANG JIAN (jian.huang@hdu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import division

import os
import cv2
import json
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils.utils import *
from utils.cutout import Cutout
from core.config_DAG import config

sample_random = random.Random()


class DAGDataset(Dataset):
    def __init__(self, cfg):
        super(DAGDataset, self).__init__()
      
        self.template_size = cfg.DAG.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.DAG.TRAIN.SEARCH_SIZE

        self.size = 25
        self.stride = cfg.DAG.TRAIN.STRIDE

        self.color = cfg.DAG.DATASET.COLOR
        self.flip = cfg.DAG.DATASET.FLIP
        self.rotation = cfg.DAG.DATASET.ROTATION
        self.blur = cfg.DAG.DATASET.BLUR
        self.shift = cfg.DAG.DATASET.SHIFT
        self.scale = cfg.DAG.DATASET.SCALE
        self.gray = cfg.DAG.DATASET.GRAY
        self.label_smooth = cfg.DAG.DATASET.LABELSMOOTH
        self.mixup = cfg.DAG.DATASET.MIXUP
        self.cutout = cfg.DAG.DATASET.CUTOUT


        self.shift_s = cfg.DAG.DATASET.SHIFTs
        self.scale_s = cfg.DAG.DATASET.SCALEs
        self.grids()
        self.neg_num = cfg.DAG.TRAIN.NEG_NUM
        self.pos_num = cfg.DAG.TRAIN.POS_NUM
        self.total_num = cfg.DAG.TRAIN.TOTAL_NUM
        self.neg = cfg.DAG.DATASET.NEG
        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
            + ([transforms.Grayscale(num_output_channels=3), ] if self.gray > random.random() else [])
            + ([Cutout(n_holes=1, length=16)] if self.cutout > random.random() else [])
        )

        print('train datas: {}'.format(cfg.DAG.TRAIN.WHICH_USE))     
        self.train_datas = []   
        start = 0
        self.num = 0    
        for data_name in cfg.DAG.TRAIN.WHICH_USE:     
            dataset = subData(cfg, data_name, start)    
            self.train_datas.append(dataset)
            start += dataset.num         
            self.num += dataset.num_use  

        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num     # 60w

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        neg = self.neg and self.neg > np.random.random()
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)
     
        if neg:
            template = dataset._get_negative_target(index)
            search = np.random.choice(self.train_datas)._get_negative_target()
        else:
            template, search = dataset._get_pairs(index, dataset.data_name)

        template, search = self.check_exists(index, dataset, template, search, neg)


        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, _, _ = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size, search=True)

        template = np.array(template)
        search = np.array(search)


        out_label = self.PointTarget(bbox, self.size, neg)

        reg_label, _ = self.reg_label(bbox, neg)
  
        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        return template, search, out_label, reg_label, np.array(bbox, np.float32)  

    # ------------------------------------
    # function groups for selecting pairs
    # ------------------------------------
    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.size      

        sz_x = sz // 2      
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),  
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2 
        self.grid_to_search_y = y * self.stride + self.search_size // 2

    def reg_label(self, bbox, neg):
        """
        generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
       
        x1, y1, x2, y2 = bbox  
        l = self.grid_to_search_x - x1  
        t = self.grid_to_search_y - y1 
        r = x2 - self.grid_to_search_x  
        b = y2 - self.grid_to_search_y  
 
        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1) 
        reg_label_min = np.min(reg_label, axis=-1)       
        inds_nonzero = (reg_label_min > 0).astype(float)  
      
        return reg_label, inds_nonzero  


    def check_exists(self, index, dataset, template, search, neg):
        name = dataset.data_name

        while True:
            if 'RGBT' in name or 'GTOT' in name and 'RGBTRGB' not in name and 'RGBTT' not in name:
                if not (os.path.exists(template[0][0]) and os.path.exists(search[0][0])):
                    index = random.randint(0, 100)
                    template, search = dataset._get_pairs(index, name)
                    continue
                else:
                    return template, search
            else:
                if not (os.path.exists(template[0]) and os.path.exists(search[0])):
                    index = random.randint(0, 100)

                    if neg :
                        
                        template = dataset._get_negative_target(index)
                        search = np.random.choice(self.train_datas)._get_negative_target()
                    else:
                        template, search = dataset._get_pairs(index, name)
                    continue
                else:
                    return template, search

    def _shuffle(self):
        """
        random shuffel
        """
        pick = []
        m = 0
        while m < self.num:     
            p = []
            for subset in self.train_datas: 
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def _get_image_anno(self, video, track, frame, RGBT_FLAG=False):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)
        if not RGBT_FLAG:
            image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
            image_anno = self.labels[video][track][frame]
            return image_path, image_anno
        else:  # rgb
            in_image_path = join(self.root, video, "{}.{}.in.x.jpg".format(frame, track))
            rgb_image_path = join(self.root, video, "{}.{}.rgb.x.jpg".format(frame, track))
            image_anno = self.labels[video][track][frame]
            in_anno = np.array(image_anno[-1][0])
            rgb_anno = np.array(image_anno[-1][1])

            return [in_image_path, rgb_image_path], (in_anno + rgb_anno) / 2

    def _get_pairs(self, index):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames)-1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        """
        like siamfc

        Args:
            image:  template or search image
            shape:  image anno

        Returns:

        """
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size  
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2   
        bbox = center2corner(Center(cx, cy, w, h)) 
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """

        draw_image = np.array(image.copy())
        x1, y1, x2, y2 = map(lambda x:int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0,255,0))
        cv2.circle(draw_image, (int(round(x1 + x2)/2), int(round(y1 + y2) /2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2)/2), int(round(y1 + y2) /2)), (int(round(x1 + x2) / 2) - 3, \
            int(round(y1 + y2) /2) -3),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imwrite(name, draw_image)

    def _draw_reg(self, image, grid_x, grid_y, reg_label, reg_weight, save_path, index):
        """
        visiualization
        reg_label: [l, t, r, b]
        """
        draw_image = image.copy()
        # count = 0
        save_name = join(save_path, '{:06d}.jpg'.format(index))
        h, w = reg_weight.shape
        for i in range(h):
            for j in range(w):
                if not reg_weight[i, j] > 0:
                    continue
                else:
                    x1 = int(grid_x[i, j] - reg_label[i, j, 0])
                    y1 = int(grid_y[i, j] - reg_label[i, j, 1])
                    x2 = int(grid_x[i, j] + reg_label[i, j, 2])
                    y2 = int(grid_y[i, j] + reg_label[i, j, 3])

                    draw_image = cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))

        cv2.imwrite(save_name, draw_image)

    def _mixupRandom(self):
        """
        gaussian random -- 0.3~0.7
        """
        return random.random() * 0.4 + 0.3

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size, search=False):
        """
        size: 127 or 255
        data augmentation for input pairs
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if search:
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = ((1.0 + self._posNegRandom() * self.scale_s), (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            #                                        4
            param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)   # shift
            param.scale = ((1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)      
        return image, bbox, param

    def _mixupShift(self, image, size):
        """
        random shift mixed-up image
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * 64, self._posNegRandom() * 64)  # shift
        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image

    # ------------------------------------
    # function for creating training label
    # ------------------------------------
    #                           
    def _dynamic_label(self, fixedLabelSize, c_shift):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift)
        return d_label
  
    def _create_dynamic_logisticloss_label(self, label_size, c_shift):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]
        sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        sz_y = sz // 2 + int(-c_shift[1] / 8)

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        c = np.where((np.abs(x) <= sz / 8), np.ones_like(y), np.zeros_like(y))
        d = np.where((np.abs(y) <= sz / 8), np.ones_like(y), np.zeros_like(y))
        e = c * d > 0

        f = np.where((np.abs(x) <= sz / 4), np.ones_like(y), np.zeros_like(y))
        g = np.where((np.abs(y) <= sz / 4), np.ones_like(y), np.zeros_like(y))
        h = f * g > 0
        label = np.where(e,
                         np.ones_like(y),               
                         np.where(h,
                                  0.5 * np.ones_like(y), 
                                  np.zeros_like(y)))     

        return label

   

    def PointTarget(self, bbox, size, neg=False):

       
        cls = -1 * np.ones((size, size), dtype=np.int64)
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num
        x1, y1, x2, y2 = bbox
        tcx, tcy, tw, th = (x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1)
        points = self.grid_to_search_x, self.grid_to_search_y

        if neg:
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4 + 1e-8) +
                           np.square(tcy - points[1]) / np.square(th / 4 + 1e-8) < 1)
            neg, neg_num = select(neg, self.neg_num)
            cls[neg] = 0

            return cls
      
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4 + 1e-8) +
                       np.square(tcy - points[1]) / np.square(th / 4 + 1e-8) < 1)
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / 2 + 1e-8) +
                       np.square(tcy - points[1]) / np.square(th / 2 + 1e-8) > 1)

        pos, pos_num = select(pos, self.pos_num)    # 16
        neg, neg_num = select(neg, self.total_num - self.pos_num)  # 48
        cls[pos] = 1
        cls[neg] = 0
        return cls

# ---------------------
# for a single dataset
# ---------------------

class subData(object):
    """
    for training with multi dataset
    """
    def __init__(self, cfg, data_name, start):
        self.data_name = data_name
        self.start = start

        info = cfg.DAG.DATASET[data_name]
        self.frame_range = info.RANGE  
        self.num_use = info.USE  
        self.root = info.PATH 

        # clean
        with open(info.ANNOTATION) as fin:
            self.labels = json.load(fin)
            self.labels = self._filter_zero(self.labels)
            self._clean()
            self.num = len(self.labels)   


        self._shuffle()   

    def _clean(self):
        """
        remove empty videos/frames/annos in dataset
        """
        # no frames
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))

        for video, track in to_del:
            try:
                del self.labels[video][track]
            except:
                pass
        to_del = []

        print(self.data_name)

        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)
        
        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
     
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:  
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]

        return self.pick     

    def _get_image_anno(self, video, track, frame):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)

        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track)) 
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno


    def _get_pairs(self, index, data_name):
        """
        get training pairs
        """
        video_name = self.videos[index]  

        video = self.labels[video_name]
        track = random.choice(list(video.keys()))   
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())   
        template_frame = random.randint(0, len(frames)-1)  # 0
        left = max(template_frame - self.frame_range, 0)  # 0
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1  # 1
        search_range = frames[left:right]
        template_frame = int(frames[template_frame])     # 0
        search_frame = int(random.choice(search_range))  # 0
        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)


    def _get_negative_target(self, index=-1):
        """
        dasiam neg
        """
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        frame = random.choice(frames)

        return self._get_image_anno(video_name, track, frame)


    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

if __name__ == '__main__':

    import os
    from torch.utils.data import DataLoader
    from core.config import config


    train_set = DAGDataset(config)
    train_loader = DataLoader(train_set, batch_size=16, num_workers=1, pin_memory=False)

    for iter, input in enumerate(train_loader):
        # label_cls = input[2].numpy()  # BCE need float
        template = input[0]
        search = input[1]
        print(template.size())
        print(search.size())


        print('dataset test')

    print()


