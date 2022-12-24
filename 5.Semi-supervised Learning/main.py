import pickle 
import pandas as pd 
import numpy as np 
import os 
from glob import glob 
import tqdm 
from PIL import Image 
from tqdm import tqdm 
import wandb

from sklearn.metrics import f1_score,accuracy_score

from src.Dataset import CifarDataset,label_unlabel_load,dataset_load
from src.Models import Model,PiModel
from src.Loss import PiCriterion

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import yaml
import warnings 
warnings.filterwarnings('ignore')

def make_transform():
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    transformer = transforms.Compose([
                                          transforms.RandomApply([color_jitter],p=0.8),
                                          transforms.RandomResizedCrop(32),
                                          transforms.GaussianBlur(kernel_size=int(0.1*32))
                                         ])
    return transformer

def make_valid(dataset = 'cifar10'):
    (train_imgs,train_labels),(_,_) = dataset_load(dataset)
    idx = np.random.choice(np.arange(len(train_imgs)),5000,replace=False)
    valid_set = {'imgs':train_imgs[idx],
        'labels':train_labels[idx]}
    return valid_set 

def train(model,criterion,optimizer,train_loader,cfg,transformer):
    global epoch 
    model.train() 
    tl_loss = [] 
    tu_loss = [] 
    total_loss = [] 
    for batch_img,batch_labels in tqdm(train_loader):
        
        batch_img_1 = transformer(batch_img.type(torch.float32).to(cfg['device']))
        batch_img_2 = transformer(batch_img.type(torch.float32).to(cfg['device']))
        batch_labels = batch_labels.to(cfg['device'])
        
        y_pred_1 = model(batch_img_1,True)
        y_pred_2 = model(batch_img_2,True)
        loss,tl,tu,weight = criterion(y_pred_1,y_pred_2,batch_labels,epoch)
        
        total_loss.append(loss.detach().cpu().numpy())
        tl_loss.append(tl.detach().cpu().numpy())
        tu_loss.append(tu.detach().cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return np.mean(total_loss),np.mean(tl_loss),np.mean(tu_loss),weight


def valid(model,test_loader,cfg):
    labels = []
    y_preds = [] 
    model.eval() 
    for batch_imgs,batch_labels in test_loader:
        batch_imgs = batch_imgs.type(torch.float32).to(cfg['device'])
        with torch.no_grad():
            y_pred = model(batch_imgs,False)
        y_pred = torch.argmax(F.softmax(y_pred),dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        
        y_preds.extend(y_pred)
        labels.extend(batch_labels.detach().cpu().numpy())    
    f1 = f1_score(np.array(y_preds),np.array(labels),average='macro')
    auc = accuracy_score(np.array(y_preds),np.array(labels))
    return f1, auc

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Exp',default=0)
    parser.add_argument('-model',default='resnet18')
    parser.add_argument('-unlabel_ratio',default=0)
    parser.add_argument('-Un_loss',default =True)
    parser.add_argument('-dataset',default='cifar10')
    args = parser.parse_args() 
    try:
        os.mkdir(f"./Save_models/{args.Exp}")
    except:
        pass 
    return args 

def yaml_save(cfg):
    with open(f"./Save_models/{cfg['Exp']}/config.yml", "w") as f:
        yaml.dump(cfg, f)
        
def exp_init(cfg):
    try:
        os.mkdir(f"./Save_models/{cfg['Exp']}")
    except:
        pass
    yaml_save(cfg)
        

if __name__ == "__main__":
    cfg = {}
    cfg['dataset'] = 'cifar10'
    cfg['model_name'] = 'resnet18'
    cfg['unlabel_ratio'] = 0
    cfg['batch_size'] = 100 
    cfg['device'] = 'cuda:0'
    cfg['lr'] = 0.003 
    cfg['beta1'] = 0.8
    cfg['beta2'] = 0.999 
    cfg['epochs'] = 300 
    cfg['std'] = 0.15  
    
    args = parse_arguments()
    cfg['Exp'] = args.Exp
    cfg['model_name'] = args.model
    cfg['unlabel_ratio'] = float(args.unlabel_ratio)
    exp_init(cfg)
#init 
    wandb.init(
                project="BA_SSL",
                name=f"{cfg['Exp']}"
            )

#Data load 
    train_set,test_set = label_unlabel_load(cfg)
    valid_set = make_valid()
    train_dataset = CifarDataset(train_set, unlabel=False)
    valid_dataset = CifarDataset(valid_set, unlabel=False)
    test_dataset = CifarDataset(test_set,unlabel=False)

    train_loader = DataLoader(train_dataset,batch_size=cfg['batch_size'],shuffle=True)
    valid_loader = DataLoader(valid_dataset,batch_size=cfg['batch_size'],shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=cfg['batch_size'],shuffle=False)

    transformer= make_transform()
#model 
    model = Model(cfg['model_name']).to('cuda')
    criterion = PiCriterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(cfg['beta1'],cfg['beta2']))

#train  
    
    best_epoch = np.inf 
    for epoch in range(cfg['epochs']):
        loss,tl_loss,tu_loss,weight =  train(model,criterion,optimizer,train_loader,cfg,transformer)
        f1 , auc = valid(model,valid_loader,cfg)
        print(f'\n Epochs : {epoch}')
        print(f'\n loss : {loss} | tl_loss : {tl_loss} | tu_loss : {tu_loss}')
        print(f'\n test f1 : {f1}')
        print(f'\n test auc : {auc}')
        
#log 
        wandb.log({ 'loss'      : loss, 
                    'tl_loss'   : tl_loss,
                    'tu_loss'   : tu_loss,
                    'weight'    : weight,
                    'f1'        : f1,
                    'auc'       : auc})
#check point             
        if loss < best_epoch:
            torch.save(model,f"./Save_models/{cfg['Exp']}/best.pt")
            best_epoch = loss 
            print(f'model saved | best loss :{best_epoch}')
