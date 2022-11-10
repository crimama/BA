import torch 
import torch.nn as nn 
import torchvision 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from tqdm import tqdm 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,Dataset
import os 
import yaml 
import wandb  

from src import Convolution_Auto_Encoder, Mnist_Dataset,MVtecADDataset,Datadir_init
from src import MVtecEncoder,MVtecDecoder,Convolution_Auto_Encoder

cfg = {'seed':42,
       'img_size':256,
       'device':'cuda:0',
       'encoded_space_dim':128,
       'lr':0.001,
       'weight_decay':1e-05,
       'batch_size':32,
       'Epochs':100,
       'target_class':6,
       'save_dir':'MVtecAD3',
       'Dataset_dir':'./Dataset/hazelnut',
       'optimizer':'adamw',
       'Decription':'Normalize제외 하고 진행'}

        


def preprocess(cfg,augmentation=None):
    #mk save dir 
    try:
        os.mkdir(f"./Save_models/{cfg['save_dir']}")
    except:
        pass
    torch.manual_seed(cfg['seed'])
    data_dir = cfg['Dataset_dir']
    Data_dir = Datadir_init()
    train_dirs = Data_dir.train_load()
    test_dirs,test_labels = Data_dir.test_load()
    indx = int(len(train_dirs)*0.8)

    train_dset = MVtecADDataset(cfg,train_dirs[:indx],Augmentation=augmentation)
    valid_dset = MVtecADDataset(cfg,train_dirs[indx:],Augmentation=augmentation)
    test_dset = MVtecADDataset(cfg,test_dirs,test_labels,Augmentation=augmentation)

    train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
    valid_loader = DataLoader(valid_dset,batch_size=cfg['batch_size'],shuffle=True)
    test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
    return train_loader,valid_loader,test_loader 

def train_epoch(model,dataloader,criterion,optimizer,scheduler,scaler):
       model.train()
       optimizer.zero_grad()
       train_loss = [] 
       for img,label in dataloader:
              img = img.to(cfg['device']).type(torch.float32)
              with torch.cuda.amp.autocast():
                    y_pred = model(img).type(torch.float32)
                    loss = criterion(img,y_pred)
              #y_pred = model(img).type(torch.float32)
              

              #Backpropagation
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update() 
              #loss.backward()
              #optimizer.step()

              #loss save 
              train_loss.append(loss.detach().cpu().numpy())
       scheduler.step() 
       print(f'\t epoch : {epoch+1} train loss : {np.mean(train_loss):.3f}')
       return np.mean(train_loss)

def valid_epoch(model,dataloader,criterion):
       model.eval()
       valid_loss = [] 
       with torch.no_grad():
              for img,label in dataloader:
                     img = img.to(cfg['device'])
                     y_pred = model(img)
                     loss = criterion(y_pred,img)
                     valid_loss.append(loss.detach().cpu().numpy())
       print(f'\t epoch : {epoch+1} valid loss : {np.mean(valid_loss):.3f}')
       #fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(5, 5))
       #ax1.imshow(img[0].detach().cpu().permute(1,2,0).numpy())
       #ax2.imshow(y_pred[0].detach().cpu().permute(1,2,0).numpy())
       #plt.show()
       return np.mean(valid_loss)    


if __name__ == "__main__":
#init 
    wandb.init(project='BA_mvtecad',name=cfg['save_dir'])
    wandb.config = cfg
    train_loader,valid_loader,test_loader   = preprocess(cfg)
#trainig intit 
    model = Convolution_Auto_Encoder(MVtecEncoder,MVtecDecoder,cfg['encoded_space_dim']).to(cfg['device'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0)
    scaler = torch.cuda.amp.GradScaler()

#Record 
    total_train_loss = [] 
    total_valid_loss = [] 
    best_valid_loss = np.inf 

    for epoch in tqdm(range(cfg['Epochs'])):
#training  
        train_loss = train_epoch(model,train_loader,criterion,optimizer,scheduler,scaler)
        valid_loss = valid_epoch(model,valid_loader,criterion)
#logging 
        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)
        wandb.log({"train_loss":train_loss})
        wandb.log({"valid_loss":valid_loss})

#check point 
        if valid_loss < best_valid_loss:
            torch.save(model,f"./Save_models/{cfg['save_dir']}/best.pt")
            best_valid_loss = valid_loss 
            print(f'\t Model save : {epoch} | best loss : {best_valid_loss :.3f}')

#prevent explosion 
        if valid_loss != valid_loss:
            model = torch.load(f"./Save_models/{cfg['save_dir']}/best.pt")
            print('Model rewinded')

#Save config 
    f = open(f"./Save_models/{cfg['save_dir']}/config.yaml",'w+')
    yaml.dump(cfg, f, allow_unicode=True)
