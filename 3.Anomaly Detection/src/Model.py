import torch.nn as nn 
import torch 

class MNistEncoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MNistEncoder,self).__init__()

        self.encoder_cnn = nn.Sequential(
                                        nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU()
                                        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
                                        nn.Linear(4*4*32,128),
                                        nn.ReLU(True),
                                        nn.Linear(128,encoded_space_dim)
                                        )
        

    def forward(self,x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x 

class MNistDecoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MNistDecoder,self).__init__()       

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim,128),
            nn.ReLU(True),
            nn.Linear(128,3*3*32),
            nn.ReLU(True)
        ) 
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32,3,3))

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,3,3,stride=2,padding=1,output_padding=1)
        )
    def forward(self,x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x 

class Convolution_Auto_Encoder(nn.Module):
    def __init__(self,Encoder,Decoder,encoded_space_dim ):
        super(Convolution_Auto_Encoder,self).__init__()
        self.encoder = Encoder(encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 


class MVtecEncoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MVtecEncoder,self).__init__()

        self.encoder_cnn = nn.Sequential(
                                        nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU()
)

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
                                        nn.Linear(8*8*128,512),
                                        nn.ReLU(True),
                                        nn.Linear(512,encoded_space_dim)
                                        )
        

    def forward(self,x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x 

class MVtecDecoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MVtecDecoder,self).__init__()       

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim,512),
            nn.ReLU(True),
            nn.Linear(512,7*7*128),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(128,7,7))      

        self.decoder_cnn = nn.Sequential(
                            nn.ConvTranspose2d(128,64,3,stride=2,output_padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(8,3,3,stride=2,padding=1,output_padding=1)
        )
    def forward(self,x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x 