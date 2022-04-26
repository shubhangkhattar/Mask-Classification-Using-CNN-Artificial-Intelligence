import torch.nn as nn

class CNN_Old(nn.Module):
    def __init__(self):
        super(CNN_Old,self).__init__()
        self.convolution_layers =  nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=(1,1)), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=128*16*16,out_features=50),
            nn.Linear(in_features=50,out_features=5)
        )

    def forward(self,X):
        X = self.convolution_layers(X)
        X =  self.fully_connected_layers(X.reshape(-1,128*16*16))
        return X
#%%
