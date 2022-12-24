import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision 
import timm 
import torch 

class Model(nn.Module):
    def __init__(self,model_name='ssl_resnet50',dataset_name='cifar10'):
        super(Model,self).__init__()
        self.model_name = model_name 
        self.encoder = self.pretrained_encoder(model_name)
        self.linear = self.output_layer(dataset_name)
        
    
    def pretrained_encoder(self,model_name):
        res = timm.create_model(model_name,pretrained=True)
        encoder = nn.Sequential(*(list(res.children())[:-1]))
        return encoder 
    
    def output_layer(self,dataset_name):
        in_features = list(self.encoder[-2][-1].children())[-3].out_channels
        if dataset_name == 'cifar10':
            return nn.Linear(in_features = in_features,out_features= 10)
        else:
            return nn.Linear(in_features = in_features,out_features= 100)
        
    def forward(self,x,_):
        x = self.encoder(x)
        x = self.linear(x)
        return x 
    
class GaussianNoise(nn.Module):
    
    def __init__(self, batch_size, input_shape=(1, 32, 32), std=0.05,device='cpu'):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape)).to(device)
        self.std = std
        
        
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise
    
    
class PiModel(nn.Module):
    def __init__(self,num_labels=10,batch_size=100,std=0.15,device='cpu'):
        super(PiModel,self).__init__()
        self.noise = GaussianNoise(batch_size,std=std,device=device)
        self.conv1 = self.conv_block(3,128).to(device)
        self.conv2 = self.conv_block(128,256).to(device)
        self.conv3 = self.conv3_block().to(device)
        self.linear = nn.Linear(128,num_labels).to(device)
        
        
    def conv_block(self,input_channel,num_filters):
        return nn.Sequential(
                                nn.Conv2d(input_channel,num_filters,3,1,1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(num_filters,num_filters,3,1,1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(num_filters,num_filters,3,1,1),
                                nn.LeakyReLU(0.1),
                                nn.MaxPool2d(2,2),
                                nn.Dropout(0.5)                    
                             )
    def conv3_block (self):
        return nn.Sequential(
                              nn.Conv2d(256,512,3,1,0),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(512,256,1,1),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(256,128,1,1),
                              nn.LeakyReLU(0.1)

        )
    def forward(self,x,train):
        if train:
            x = self.noise(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.avg_pool2d(x, x.size()[2:]).squeeze()
        x = self.linear(x)
        return x 