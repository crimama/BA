import torch 
from torch.utils.data import Dataset 
import torchvision.transforms as transforms 
from glob import glob 
import numpy as np 
import pickle 
class CifarDataset(Dataset):
    def __init__(self,data,unlabel=False,transform=None):
        super(CifarDataset,self).__init__()
        self.transform = transform 
        self.imgs = data['imgs']
        self.labels = data['labels']
        self.unlabel = unlabel 
        self.transform = self.transfrom_init(transform)
        
    def __len__(self):
        return len(self.imgs)
    
    def transfrom_init(self,transform):
        if transform == None:
            return transforms.Compose([transforms.ToTensor()])
        else:
            return transform 

            
    def __getitem__(self,idx):
        if self.unlabel:
            img = self.transform(self.imgs[idx])
            return img
        else:
            img = self.transform(self.imgs[idx])
            label = self.labels[idx]
            return img,label 
        
def from_pickle_to_img(file,name):
    with open(file, 'rb') as f:
        data = pickle.load(f,encoding='bytes')
    if name == 'cifar10':
        batch_imgs = data[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
        batch_labels = data[b'labels']
        return batch_imgs, np.array(batch_labels) 
    
    elif name == 'cifar100':
        batch_imgs = data[b'data'].reshape(-1,32,32,3)
        batch_labels = data[b'fine_labels']
        return batch_imgs, np.array(batch_labels) 

def load_cifar10():
    files = sorted(glob('./Dataset/cifar-10-batches-py/*')[1:-1])
    imgs = [] 
    labels = [] 
    for file in files:
        batch_imgs,batch_labels = from_pickle_to_img(file,'cifar10')
        imgs.extend(batch_imgs)
        labels.extend(batch_labels)
    labels = np.array(labels)
    imgs = np.array(imgs)
    return (imgs[:50000],labels[:50000]),(imgs[50000:],labels[50000:])

def load_cifar100():
    files = sorted(glob('./Dataset/cifar-100-python/*'))[-2:]
    train_imgs,train_labels = from_pickle_to_img(files[1],'cifar100')
    test_imgs,test_labels = from_pickle_to_img(files[0],'cifar100')
    return (train_imgs,train_labels),(test_imgs,test_labels)
        
    
def img_load_all(datasets = None):
    if datasets == 'cifar10':
        return load_cifar10()
    elif datasets == 'cifar100':
        return load_cifar100()

'''
def label_unlabel_load(cfg):
        (train_imgs,train_labels),(test_imgs,test_labels) = img_load_all(cfg['dataset'])
        split_std = int(len(train_imgs)*(1-cfg['unlabel_ratio']))
        train_label = {'imgs':train_imgs[:split_std],
                'labels':train_labels[:split_std]} 
        train_unlabel = {'imgs':train_imgs[split_std:],
                        'labels':None} 
        test = {'imgs':test_imgs,
                'labels':test_labels}
        return train_label,train_unlabel,test    
''' 
def label_unlabel_load(cfg):
    (train_imgs,train_labels),(test_imgs,test_labels) = img_load_all(cfg['dataset'])
    labels = np.unique(train_labels)
    label = labels[0]
    for label in labels:
        label_idx = (train_labels ==label).nonzero()[0]
        unlabel_idx = np.random.choice(label_idx,int(len(label_idx)*cfg['unlabel_ratio']),replace=False)
        train_labels[unlabel_idx] = -1 
        
    train = {'imgs':train_imgs,
            'labels':train_labels}
    test = {'imgs':test_imgs,
            'labels':test_labels}
    return train, test 