import medpy
from medpy.io import load
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import numpy as np
class MyDataset(Dataset):
    def __init__(self, root_l, subfolder_l,root_s,subfolder_s,prefixs, transform=None):
        super(MyDataset, self).__init__()
        self.prefixs=prefixs
        self.l_path = os.path.join(root_l, subfolder_l)
        self.s_path=os.path.join(root_s, subfolder_s)
        self.templ = [x for x in os.listdir(self.l_path) if os.path.splitext(x)[1] == ".img"]
        self.temps = [x for x in os.listdir(self.s_path) if os.path.splitext(x)[1] == ".img"]
        self.image_list_l=[]
        self.image_list_s = []
        #找指定前缀的数据
        for file in self.templ:
            for pre in prefixs:
                if pre in file:
                    self.image_list_l.append(file)
        #找指定前缀的数据
        for file in self.temps:
            for pre in prefixs:
                if pre in file:
                    self.image_list_s.append(file)
        # print(self.image_list_l)
        # print(self.image_list_s)
        self.transform = transform

    def __len__(self):
        return len(self.image_list_l)

    def __getitem__(self, item):
        #读图片（低剂量PET）
        image_path_l = os.path.join(self.l_path, self.image_list_l[item])
        #image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        image_l,h=load(image_path_l)
        image=np.array(image_l)
        #print(image.shape)
        if self.transform is not None:
            image = self.transform(image_l)
        #读标签(高质量PET)
        image_path_s = os.path.join(self.s_path, self.image_list_s[item])
        image_s,h2=load(image_path_s)
        image_s=np.array(image_s)
        #print(image_l.shape)0
        # print(image_path_l,image_path_s)
        #添加通道维度
        image_l=image_l[np.newaxis,:]
        image_s=image_s[np.newaxis,:]
        image_l=torch.Tensor(image_l)
        image_s=torch.Tensor(image_s)
        #print(image.shape)
        if self.transform is not None:
            image = self.transform(image_s)
        #返回：影像，标签
        return image_l, image_s
###
class MyMultiDataset(Dataset):
    def __init__(self, root_l, subfolder_l,root_s,subfolder_s,root_mri,subfolder_mri,prefixs, transform=None):
        super(MyMultiDataset, self).__init__()
        self.prefixs=prefixs
        self.l_path = os.path.join(root_l, subfolder_l)
        self.s_path=os.path.join(root_s, subfolder_s)
        self.templ = [x for x in os.listdir(self.l_path) if os.path.splitext(x)[1] == ".img"]
        self.temps = [x for x in os.listdir(self.s_path) if os.path.splitext(x)[1] == ".img"]
        self.image_list_l=[]
        self.image_list_s = []
        self.image_list_mri = []
        #找指定前缀的数据
        for file in self.templ:
            for pre in prefixs:
                if pre in file:
                    self.image_list_l.append(file)
        #找指定前缀的数据
        for file in self.temps:
            for pre in prefixs:
                if pre in file:
                    self.image_list_s.append(file)
        #找指定前缀的数据
        for file in self.temp_mri:
            for pre in prefixs:
                if pre in file:
                    self.image_list_mri.append(file)
        # print(self.image_list_l)
        # print(self.image_list_s)
        self.transform = transform

    def __len__(self):
        return len(self.image_list_l)

    def __getitem__(self, item):
        #读图片（低剂量PET）
        image_path_l = os.path.join(self.l_path, self.image_list_l[item])
        #image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        image_l,h=load(image_path_l)
        image=np.array(image_l)
        #print(image.shape)
        if self.transform is not None:
            image = self.transform(image_l)
        #读标签(高质量PET)
        image_path_s = os.path.join(self.s_path, self.image_list_s[item])
        image_s,h2=load(image_path_s)
        image_s=np.array(image_s)
        #print(image_l.shape)0
        # print(image_path_l,image_path_s)
        #添加通道维度
        image_l=image_l[np.newaxis,:]
        image_s=image_s[np.newaxis,:]
        image_l=torch.Tensor(image_l)
        image_s=torch.Tensor(image_s)
        #print(image.shape)
        if self.transform is not None:
            image = self.transform(image_s)
        #返回：影像，标签
        return image_l, image_s
#
#data
def loadData(root1, subfolder1,root2,subfolder2,prefixs, batch_size, shuffle=True):

    transform = None
    #测试已修改
    dataset = MyDataset(root1, subfolder1,root2,subfolder2,prefixs,transform=transform)
    #dataset = MyDataset(root, subfolder,transform=None)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#multi data
def loadMultiData(root1, subfolder1,root2,subfolder2,root3,subfolder3,prefixs, batch_size, shuffle=True):

    transform = None
    #测试已修改
    dataset = MyMultiDataset(root1, subfolder1,root2,subfolder2,root3,subfolder3,prefixs,transform=transform)
    #dataset = MyDataset(root, subfolder,transform=None)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
#x=MyDataset('./data/l_cut','')
def readTxtLineAsList(txt_path):
    fi = open(txt_path, 'r')
    txt = fi.readlines()
    res_list = []
    for w in txt:
        w = w.replace('\n', '')
        res_list.append(w)
    return res_list

if __name__ == '__main__':
    train_txt_path = r"E:\Projects\PyCharm Projects\dataset\split\Ex2\train.txt"
    val_txt_path = r"E:\Projects\PyCharm Projects\dataset\split\Ex2\val.txt"
    train_imgs = readTxtLineAsList(train_txt_path)
    print(train_imgs)
    val_imgs = readTxtLineAsList(val_txt_path)
    print(val_imgs)
    trainloader=loadData('E:\Projects\PyCharm Projects\dataset\clinical/train_l_cut','','E:\Projects\PyCharm Projects\dataset\clinical/train_s_cut','',prefixs=train_imgs,batch_size=1)
    valloader = loadData('E:\Projects\PyCharm Projects\dataset\clinical/train_l_cut', '',
                           'E:\Projects\PyCharm Projects\dataset\clinical/train_s_cut', '', prefixs=val_imgs,
                           batch_size=1)