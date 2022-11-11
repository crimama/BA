import warnings 
warnings.filterwarnings('ignore')
import torch 
 

import numpy as np 
import matplotlib.pyplot as plt 


import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from src import MVtecADDataset,Datadir_init




class Reconstruction_Metric:
    def __init__(self,cfg,augmentation=None):
        self.cfg = cfg 
        self.train_loader,self.test_loader = self._preprocess(cfg,augmentation)
        self.model = torch.load(f"./Save_models/{cfg['save_dir']}/best.pt").to(cfg['device'])

    def _preprocess(self,cfg,augmentation):
        torch.manual_seed(cfg['seed'])
        data_dir = cfg['Dataset_dir']
        Data_dir = Datadir_init(data_dir)
        train_dirs = Data_dir.train_load()
        test_dirs,test_labels = Data_dir.test_load()

        train_dset = MVtecADDataset(cfg,train_dirs,Augmentation=augmentation)
        test_dset = MVtecADDataset(cfg,test_dirs,test_labels,Augmentation=augmentation)

        train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
        test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
        return train_loader,test_loader 

    def _test_data_inference(self):
        Pred_imgs = [] 
        True_imgs = [] 
        True_labels = [] 
        self.model.eval()
        for img,label in self.test_loader:
            img = img.to(self.cfg['device']).type(torch.float32)

            with torch.no_grad():
                Pred_img = self.model(img)
            
            Pred_imgs.extend(Pred_img.detach().cpu().numpy())
            True_imgs.extend(img.detach().cpu().numpy())
            True_labels.extend(label.detach().numpy())

        Pred_imgs = np.array(Pred_imgs)
        True_imgs = np.array(True_imgs)
        True_labels = np.array(True_labels)
        True_labels = np.where(True_labels==0,True_labels,1) # Anomaly:1, normal:0 `
        test_score = np.mean(np.array(Pred_imgs-True_imgs).reshape(len(True_labels),-1)**2,axis=1)
        return test_score, True_labels

    def _train_data_inference(self):
        Train_Pred_imgs = [] 
        Train_True_imgs = [] 
        for img,_ in self.train_loader:
            img = img.to(self.cfg['device']).type(torch.float32)
            
            with torch.no_grad():
                Pred_img =  self.model(img)
            Train_Pred_imgs.extend(Pred_img.detach().cpu().numpy())
            Train_True_imgs.extend(img.detach().cpu().numpy())

        Train_Pred_imgs = np.array(Train_Pred_imgs)
        Train_True_imgs = np.array(Train_True_imgs)

        Train_score = np.mean(np.array(Train_Pred_imgs-Train_True_imgs).reshape(len(Train_True_imgs),-1)**2,axis=1)
        return Train_score 

    def Metric_auroc_roc(self,Test_score,True_labels,plot=0):
        fpr,tpr,threshold = roc_curve(True_labels,Test_score,pos_label=0)
        AUROC = round(auc(fpr, tpr),4)

        if plot != 0:
            plt.plot(fpr,tpr)
            plt.title("ROC curve")
            plt.show()
        
        return AUROC,[fpr.tolist(),tpr.tolist(),threshold.tolist()]

    def Metric_Score(self,Train_score,Test_score,True_labels):
        Threshold = np.percentile(Train_score,80)
        Pred_labels = np.array(Test_score>Threshold).astype(int)
        ACC = accuracy_score(True_labels,Pred_labels)
        PRE = precision_score(True_labels,Pred_labels)
        RECALL = recall_score(True_labels,Pred_labels)
        F1 = f1_score(True_labels,Pred_labels)
        return ACC,PRE,RECALL,F1
        

    def main(self):
        Test_score, True_labels = self._test_data_inference()
        Train_score  =  self._train_data_inference()
        AUROC, ROC = self.Metric_auroc_roc(Test_score,True_labels)
        ACC,PRE,RECALL,F1 = self.Metric_Score(Train_score,Test_score,True_labels)

        return [AUROC,ROC], [ACC,PRE,RECALL,F1]



class Machine_Metric:
    def __init__(self,cfg,augmentation=None):
        self.cfg = cfg 
        self.train_loader,self.test_loader = self._preprocess(cfg,augmentation)
        self.model = torch.load(f"./Save_models/{cfg['save_dir']}/best.pt").to(cfg['device'])
        self.encoder = self.model.encoder 

    def _preprocess(self,cfg,augmentation):
        torch.manual_seed(cfg['seed'])
        data_dir = cfg['Dataset_dir']
        Data_dir = Datadir_init(data_dir)
        train_dirs = Data_dir.train_load()
        test_dirs,test_labels = Data_dir.test_load()

        train_dset = MVtecADDataset(cfg,train_dirs,Augmentation=augmentation)
        test_dset = MVtecADDataset(cfg,test_dirs,test_labels,Augmentation=augmentation)

        train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
        test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
        return train_loader,test_loader 

    def _train_data_inference(self):
        normal_vecs = [] 
        normal_labels = [] 
        
        #Inference 
        for img,label in self.train_loader:
            img = img.to(self.cfg['device']).type(torch.float32)
            
            with torch.no_grad():
                normal_vec =  self.encoder(img)
                
            normal_vecs.extend(normal_vec.detach().cpu().numpy())
            normal_labels.extend(label.detach().cpu().numpy())
            
        normal_vecs = np.array(normal_vecs)
        normal_labels = np.array(normal_labels)

        return normal_vecs,normal_labels 

    def _test_data_inference(self):
        test_vecs = [] 
        test_labels = [] 
        for img,label in self.test_loader:
            img = img.to(self.cfg['device']).type(torch.float32)

            with torch.no_grad():
                test_vec = self.encoder(img)
            test_vecs.extend(test_vec.detach().cpu().numpy())
            test_labels.extend(label.detach().cpu().numpy())
        test_vecs = np.array(test_vecs)
        test_labels = np.array(test_labels)
        test_labels = np.where(test_labels==0,test_labels,-1) # Anomaly:-1, normal:0
        test_labels = np.where(test_labels==1,test_labels,1) # Anomaly:-1, normal:1

        return test_vecs,test_labels 
    def _Metric_score(self,True_labels,Pred_labels):
        ACC = accuracy_score(True_labels,Pred_labels)
        PRE = precision_score(True_labels,Pred_labels)
        RECALL = recall_score(True_labels,Pred_labels)
        F1 = f1_score(True_labels,Pred_labels)
        return ACC,PRE,RECALL,F1

    def scaling(self,normal_vecs):
        self.minmax = MinMaxScaler()
        normalized_vecs = self.minmax.fit_transform(normal_vecs)
        return normalized_vecs

    def main(self):
        normal_vecs,normal_labels = self._train_data_inference()
        test_vecs , test_labels = self._test_data_inference()
        normalized_vecs = self.scaling(normal_vecs)
        normalized_test_vecs = self.minmax.transform(test_vecs)


        self.model = OneClassSVM()
        self.model.fit(normalized_vecs)
        Pred_labels = self.model.predict(normalized_test_vecs)

        ACC,PRE,RECALL,F1 = self._Metric_score(test_labels,Pred_labels)
        return ACC,PRE,RECALL,F1

        