import torch.nn as nn 
import torch 
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
import json
import argparse 

from src import Convolution_Auto_Encoder, Mnist_Dataset,MVtecADDataset,Datadir_init
from src import MVtecEncoder,MVtecDecoder,Convolution_Auto_Encoder
from src import Machine_Metric,Reconstruction_Metric
from src import create_transformation
from src.set_transformation import create_transformation



def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-aug_number')
    parser.add_argument('-save_dir')
    args = parser.parse_args() 
    return args 


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
    valid_dset = MVtecADDataset(cfg,train_dirs[indx:])
    test_dset = MVtecADDataset(cfg,test_dirs,test_labels)

    train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
    valid_loader = DataLoader(valid_dset,batch_size=cfg['batch_size'],shuffle=False)
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
                     loss = torch.sqrt(criterion(y_pred,img))
                     valid_loss.append(loss.detach().cpu().numpy())
       print(f'\t epoch : {epoch+1} valid loss : {np.mean(valid_loss):.3f}')
       fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(5, 5))
       ax1.imshow(img[0].detach().cpu().permute(1,2,0).numpy())
       ax2.imshow(y_pred[0].detach().cpu().permute(1,2,0).numpy())
       plt.show()
       return np.mean(valid_loss)   

def Save_result(cfg):

    f = open(f"./Save_models/{cfg['save_dir']}/config.yaml",'w+')
    yaml.dump(cfg, f, allow_unicode=True)
    
    metric =  {} 
    metric['auto'] = {} 
    metric['machine']={}

    machine = Machine_Metric(cfg)
    auto = Reconstruction_Metric(cfg)

    [auroc,roc], [acc,pre,rec,f1] = machine.main()
    [AUROC,ROC], [ACC,PRE,RECALL,F1] = auto.main() 

    metric['auto']['auroc'] = AUROC
    metric['auto']['roc'] =  ROC 
    metric['auto']['metric'] =[ACC,PRE,RECALL,F1]
    metric['machine']['roc'] = roc 
    metric['machine']['auroc']=auroc 
    metric['machine']['metric']=[acc,pre,rec,f1]


    with open(f"./Save_models/{cfg['save_dir']}/Metric.json",'w') as f:
       json.dump(metric,f)

    return metric 


if __name__ == "__main__":
    args = parse_arguments()
#init 
    cfg = yaml.load(open('./init_config.yaml','r'), Loader=yaml.FullLoader)
    cfg['save_dir'] = args.save_dir
    cfg['aug_number'] = int(args.aug_number)

    trans = create_transformation(cfg)
    wandb.init(project='BA_MVtec2',name=cfg['save_dir'])
    wandb.config = cfg
    train_loader,valid_loader,test_loader   = preprocess(cfg,trans)
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
    print('Training start')
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
    print('Training Done')

    metric = Save_result(cfg)
    print('Metric Done')
