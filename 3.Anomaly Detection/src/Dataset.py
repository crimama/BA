from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from PIL import Image 
import torch 
import cv2 
from glob import glob 
import pandas as pd 
import numpy as np 


class Mnist_Dataset(Dataset):
    def __init__(self,dataset,transforms=None,target_class=None):
        super(Mnist_Dataset,self).__init__()
        self.dataset = dataset 

        self.target_class = target_class 
        self.total_imgs,self.labels = self.filter_class_img(dataset)
        
        self.transforms = transforms 
        
        

    def filter_class_img(self,dataset):
        if self.target_class == None:
            return dataset.data, dataset.targets 
        else:
            target_indx = torch.where(dataset.targets==self.target_class)[0]
            imgs = dataset.data[target_indx]
            labels = dataset.targets[target_indx]
            return imgs,labels 


    def __getitem__(self,idx):
        img = self.total_imgs[idx].type(torch.float).reshape(1,28,28)
        label = self.labels[idx]

        if self.transforms != None:
            img = self.transforms(img)

        return img/255., label 


    def __len__(self):
        return len(self.total_imgs)


class MVtecADDataset(Dataset):
    def __init__(self,cfg,img_dirs,labels=None,Augmentation=None):
        super(MVtecADDataset,self).__init__()
        self.cfg = cfg 
        self.dirs = img_dirs 
        self.augmentation = self.__init_aug__(Augmentation)
        self.labels = self.__init_labels__(labels)

    def __len__(self):
        return len(self.dirs)

    def __init_labels__(self,labels):
        if np.sum(labels) !=None:
            return labels 
        else:
            return np.zeros(len(self.dirs))
    
    def __init_aug__(self,Augmentation):
        if Augmentation == None:
            augmentation = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize((self.cfg['img_size'],self.cfg['img_size']))
                                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                            ])
        else: 
            augmentation = Augmentation 
        return augmentation                                      

    def __getitem__(self,idx):
        img_dir = self.dirs[idx]
        img = Image.open(img_dir)
        img = self.augmentation(img)

        if np.sum(self.labels) !=None:
            return img,self.labels[idx] 
        else:
            return img


class Datadir_init:
    def __init__(self,Dataset_dir='./Dataset/hazelnut'):
        self.Dataset_dir = Dataset_dir 
    def test_load(self):
        test_label_unique = pd.Series(sorted(glob(f'{self.Dataset_dir}/test/*'))).apply(lambda x : x.split('/')[-1]).values
        test_label_unique = {key:value for value,key in enumerate(test_label_unique)}
        self.test_label_unique = test_label_unique 

        test_dir = f'{self.Dataset_dir}/test/'
        label = list(test_label_unique.keys())[0]

        test_img_dirs = [] 
        test_img_labels = [] 
        for label in list(test_label_unique.keys()):
            img_dir = sorted(glob(test_dir +f'{label}/*'))
            img_label = np.full(len(img_dir),test_label_unique[label])
            test_img_dirs.extend(img_dir)
            test_img_labels.extend(img_label)
        return np.array(test_img_dirs),np.array(test_img_labels) 

    def train_load(self):
        train_img_dirs = sorted(glob(f'{self.Dataset_dir}/train/good/*.png'))
        return np.array(train_img_dirs) 