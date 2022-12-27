import torch.nn as nn 
import torch 
import math 
import numpy as np 
class PiCriterion:
    def __init__(self,cfg):
        self.label_criterion = nn.CrossEntropyLoss()
        self.unlabel_criterion = nn.MSELoss()
        self.n_labeled = 50000 * (1-cfg['unlabel_ratio'])
        self.superonly = cfg['super_only']
        
    def ramp_up(self,epoch, max_epochs=80, max_val=30, mult=-5):
        if epoch == 0:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


    def weight_schedule(self,epoch, max_epochs=80, max_val=30,
                        mult=-5, n_samples=50000):
        
            max_val = max_val * (float(self.n_labeled) / n_samples)
            return self.ramp_up(epoch, max_epochs, max_val, mult)
        
    def __call__(self,y_pred_1,y_pred_2,batch_labels,epoch):
        label_idx = (batch_labels!=-1).nonzero().flatten()
        #supervised loss 
        tl_loss = self.label_criterion(y_pred_1[label_idx],
                                       batch_labels[label_idx])
        #unsupervised loss 
        weight = self.weight_schedule(epoch)
        weight = torch.autograd.Variable(torch.FloatTensor([weight]).cuda(), requires_grad=False)
        tu_loss = self.unlabel_criterion(y_pred_1,y_pred_2) * weight 
        
        if self.superonly:
            return tl_loss, tl_loss ,torch.tensor([0]),torch.tensor([0])
        else:
            #total loss 
            loss = tl_loss + tu_loss  
            
            return loss, tl_loss, tu_loss, weight 
            
        