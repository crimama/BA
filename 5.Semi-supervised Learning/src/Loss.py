import torch.nn as nn 
import math 
import numpy as np 
class PiCriterion:
    def __init__(self):
        self.label_criterion = nn.CrossEntropyLoss()
        self.unlabel_criterion = nn.MSELoss()
        
    def ramp_up_function(self,epoch, epoch_with_max_rampup=80):

        if epoch < epoch_with_max_rampup:
            p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
            p = 1.0 - p
            return math.exp(-p*p*5.0)
        else:
            return 1.0
    def ramp_up(self,epoch, max_epochs=80, max_val=30, mult=-5):
        if epoch == 0:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


    def weight_schedule(self,epoch, max_epochs=80, max_val=30, mult=-5, n_labeled=60000, n_samples=60000):
            max_val = max_val * (float(n_labeled) / n_samples)
            return self.ramp_up(epoch, max_epochs, max_val, mult)
        
    def __call__(self,y_pred_1,y_pred_2,batch_labels,epoch):
        label_idx = (batch_labels!=-1).nonzero()
                

        #supervised loss 
        tl_loss = self.label_criterion(y_pred_1[label_idx].squeeze(),batch_labels[label_idx].flatten())
        #unsupervised loss 
        tu_loss = self.unlabel_criterion(y_pred_1,y_pred_2) * self.weight_schedule(epoch)
        self.tu_loss = tu_loss
        loss = tl_loss + tu_loss  
        return loss, tl_loss, tu_loss 
        