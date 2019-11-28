##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## Modified by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import cv2
import pdb
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from PIL import Image, ImageOps, ImageFilter
import random
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms


class LeadBangTrain(data.Dataset):
    def __init__(self, root, max_iters=None,
        scale=True, mirror=True, ignore_label=255, use_aug=False, network="resnet101"):
        self.root = root
        # self.crop_h, self.crop_w = crop_size
        self.crop_h = 480
        self.crop_w = 480
        self.img_width = 512
        self.img_height = 512
        self.scale = scale
        self.ignore_label = ignore_label     
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.network = network
        self.files = []
        self.cache_img = {}
        self.cache_label = {}
        self.item_idx_list = []
        for item_idx in range(1, 1463):
            self.item_idx_list.append(item_idx)
            img_path = 'source/' + str(item_idx) + ".bmp"
            label_path = 'label/' + str(item_idx) + ".bmp"
            img_file = osp.join(self.root, img_path)
            label_file = osp.join(self.root, label_path)
            print('label file: ', label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": str(item_idx),
                "weight": 1
            })
            self.cache_img[item_idx] = cv2.imread(img_file)
            self.cache_label[item_idx] = 255 - cv2.imread(label_file, 0)
        
        print('{} images are loaded!'.format(1462))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label
    
    def rescale(self, image, label):
        image = cv2.resize(image, (self.img_width, self.img_height))
        label = cv2.resize(label, (self.img_width, self.img_height))
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def get_rotate_angle(self, angle_min, angle_max, angle_delta):
        count = int((angle_max - angle_min) / angle_delta)
        delta = random.random() * (count + 1) * angle_delta
        angle = angle_min + delta
        if angle < angle_min:
            angle = angle_min
        if angle > angle_max:
            angle = angle_max
        return angle

    def rotate(self, image, angle, border_value=None): 
        center = (self.img_width // 2, self.img_height // 2)
    
        M = cv2.getRotationMatrix2D(center, angle, 1)
        if border_value is None:
            rotated = cv2.warpAffine(image, M, (self.img_width, self.img_height))
        else:
            rotated = cv2.warpAffine(image, M, (self.img_width, self.img_height), borderValue=(int(border_value),))
        return rotated
    def get_border_value(self, mat):
        r = mat.shape[0]
        c = mat.shape[1]
        return (mat[1][1] + mat[1][c - 2] + mat[r-2][1] + mat[r-2][c-2] + mat[2][2] + mat[2][c - 3] + mat[r-3][2] + mat[r-3][c-3]) / 8.0
    def rotate_img_lb(self, image, label, angle):
        b = image[0]
        g = image[1]
        r = image[2]
        # (102.9801, 115.9465, 122.7717)
        # b = self.rotate(b, angle, border_value=255 - 102.9801)
        # g = self.rotate(g, angle, border_value=255-115.9465)
        # r = self.rotate(r, angle, border_value=255-122.7717)
        b = self.rotate(b, angle, border_value=self.get_border_value(b))
        g = self.rotate(g, angle, border_value=self.get_border_value(g))
        r = self.rotate(r, angle, border_value=self.get_border_value(r))
        label = self.rotate(label, angle)
        image = np.asarray([b, g, r], dtype=np.float32)
        ret, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        return image, label

    def adv_img_lb(self, img, lb):

        # brightness
        img += (random.random() * 10 - 5)

        # rotate
        angle = self.get_rotate_angle(-180, 180, 5)
        img, lb = self.rotate_img_lb(img, lb, angle)
        
        # flip lr
        if random.random() < 0.5:
            img = img[:,:,::-1]
            lb = lb[:,::-1]
        # flip ud
        if random.random() < 0.5:
            img = img[:,::-1,:]
            lb = lb[::-1,:]

        return img, lb

    def __getitem__(self, index):
        datafiles = self.files[index]
        item_idx = self.item_idx_list[index]
        image = self.cache_img[item_idx].copy()
        label = self.cache_label[item_idx].copy()

        size = image.shape
        name = datafiles["name"]
        image, label = self.rescale(image, label)

        image = np.array(image, dtype=np.float32)

        # print("network: {}, mean: {}, var: {}".format(self.network, mean, var))
        image = image[:,:,::-1]

        # elif self.network == "wide_resnet38":
        #     mean = (0.41738699, 0.45732192, 0.46886091)
        #     var = (0.25685097, 0.26509955, 0.29067996)
        #     image = image[:,:,::-1]
        #     image /= 255
        #     image -= mean   
        #     image /= var
        image = image.transpose((2, 0, 1))
        image, label = self.adv_img_lb(image, label)
        
        # bgr
        # mean:  [0.5590945  0.4735202  0.67869484]
        # var:  [0.15844876 0.16141151 0.09829281]
        # rgb
        image = image.transpose((1, 2, 0))
        mean = (0.67869484, 0.4735202, 0.5590945)
        var = (0.09829281, 0.16141151, 0.15844876)
        image /= 255
        image -= mean
        image /= var
        image = image.transpose((2, 0, 1))

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[:,h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        # [0, 255] => [0, 1]
        ret, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)
        label = np.array(label, dtype=np.int64)
        return image.copy(), label.copy(), np.array(size), name


class LeadBangTest(data.Dataset):
    def __init__(self, root, max_iters=None,
        scale=True, mirror=True, ignore_label=255, network="resnet101"):
        self.root = root
        # self.crop_h, self.crop_w = crop_size
        self.crop_h = 512
        self.crop_w = 512
        self.img_width = 512
        self.img_height = 512
        self.scale = scale
        self.ignore_label = ignore_label     
        self.is_mirror = mirror
        self.network = network
        self.files = []
        self.cache_img = {}
        self.cache_label = {}
        self.item_idx_list = []
        for item_idx in range(1463, 2352):
            self.item_idx_list.append(item_idx)
            img_path = 'source/' + str(item_idx) + ".bmp"
            label_path = 'label/' + str(item_idx) + ".bmp"
            img_file = osp.join(self.root, img_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": str(item_idx),
                "weight": 1
            })
            print("label: ", label_file)
            self.cache_img[item_idx] = cv2.imread(img_file)
            self.cache_label[item_idx] = 255-cv2.imread(label_file, 0)
        
        print('{} images are loaded!'.format(2352-1463))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def rescale(self, image, label):
        image = cv2.resize(image, (self.img_width, self.img_height))
        label = cv2.resize(label, (self.img_width, self.img_height))
        return image, label
    def __getitem__(self, index):
        datafiles = self.files[index]
        item_idx = self.item_idx_list[index]
        image = self.cache_img[item_idx].copy()
        label = self.cache_label[item_idx].copy()

        size = image.shape
        name = datafiles["name"]
        image, label = self.rescale(image, label)

        image = np.array(image, dtype=np.float32)

        # if self.network == "resnet101":
        #     # mean = (102.9801, 115.9465, 122.7717)
        #     mean = (172.43679809570312, 123.58832550048828, 141.29403686523437)

        #     image = image[:,:,::-1]
        #     image -= mean
        # elif self.network == "mobilenetv2":

        # bgr
        # mean:  [0.5590945  0.4735202  0.67869484]
        # var:  [0.15844876 0.16141151 0.09829281]
        # rgb
        mean = (0.67869484, 0.4735202, 0.5590945)
        var = (0.09829281, 0.16141151, 0.15844876)
        # print("network: {}, mean: {}, var: {}".format(self.network, mean, var))
        image = image[:,:,::-1]
        image /= 255
        image -= mean
        image /= var

        # elif self.network == "wide_resnet38":
        #     mean = (0.41738699, 0.45732192, 0.46886091)
        #     var = (0.25685097, 0.26509955, 0.29067996)
        #     image = image[:,:,::-1]
        #     image /= 255
        #     image -= mean   
        #     image /= var
        image = image.transpose((2, 0, 1))

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[:,h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        # [0, 255] => [0, 1]
        ret, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)
        label = np.array(label, dtype=np.int64)
        label = np.asarray(label, dtype=np.int64)    
        return image.copy(), label.copy(), np.array(size), name


def test_train_leadbang(path):
    
    dst = LeadBangTrain(path)
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=0)

    for i, dt in enumerate(trainloader):
        imgs, labels, _, name = dt
        img = imgs.numpy()
        lb = labels.numpy()
        print(name)
        print(img.shape)
        print(lb.shape)
        name = name[0]
        img = np.transpose(img[0], (1,2,0))
        img *= (0.09829281, 0.16141151, 0.15844876)
        img += (0.67869484, 0.4735202, 0.5590945)
        img *= 255
        img = img[:,:,::-1]
        img = np.array(img, dtype=np.uint8)
        lb = 255 - lb[0] * 255
        lb = np.asarray(lb, dtype=np.uint8)
        cv2.imshow( "img", img)
        cv2.imshow( "lb", lb)
        cv2.waitKey(0)


def test_test_leadbang(fpath):
    
    dst = LeadBangTest(fpath)
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=0)

    for i, dt in enumerate(trainloader):
        imgs, labels, _, name = dt
        img = imgs.numpy()
        lb = labels.numpy()
        print(name)
        print(img.shape)
        print(lb.shape)
        name = name[0]
        img = np.transpose(img[0], (1,2,0))
        img *= (0.09829281, 0.16141151, 0.15844876)
        img += (0.67869484, 0.4735202, 0.5590945)
        img *= 255
        img = img[:,:,::-1]
        img = np.array(img, dtype=np.uint8)
        lb = 255 - lb[0] * 255
        lb = np.asarray(lb, dtype=np.uint8)
        cv2.imshow( "img", img)
        cv2.imshow( "lb", lb)
        cv2.waitKey(0)

def compute_mean_var(path):
    img_list = os.listdir(path)
    a = []
    mat_list = []
    for img_name in img_list:
        if img_name[-4:] != '.bmp':
            continue
        fpath = path + '/' + img_name
        img = cv2.imread(fpath)
        img = np.array(img, dtype=np.float32)
        img /= 255.0
        mat_list.append(img.copy())
        img = np.reshape(img, (-1, 3))
        m = np.mean(img, axis=0)
        a.append(m.copy())
    a = np.array(a, dtype=np.float32)
    a= np.mean(a, axis=0)
    print('mean: ', a)
    b = []
    for mat in mat_list:
        dmat = mat - a
        dmat2 = dmat * dmat
        dmat2 = np.reshape(dmat2, (-1, 3))
        var = np.mean(dmat2, axis=0)
        b.append(var.copy())
    b = np.array(b, dtype=np.float32)
    b = np.mean(b, axis=0)
    print('var: ', b)

if __name__ == '__main__':
    # test_train_leadbang("C:/Users/86180/work/segment/OCNet.pytorch/dataset/leadbang/")
    test_test_leadbang("C:/Users/86180/work/segment/OCNet.pytorch/dataset/leadbang/")
    # compute_mean_var("C:/Users/86180/work/segment/OCNet.pytorch/dataset/leadbang/source")
