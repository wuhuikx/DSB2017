import torch
from torch import nn



class ResNet(nn.Module):
    def __init__(self):
        self.preBlock = nn.Sequential(
            nn.Conv3d(1,24,kernel_size=3,pad=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24,24,kernel_size=3,pad=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))
        self.ReLU = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3D(kernel_size = 2, stride =2, return_indices = True)
        #self.unmaxpool = nn.MaxUnPool3d(kernel_size=2, stride = 2)
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        
        self.forward1_1 = mainBlock(24, 32, 3, 1)
        self.shortcut1_1 = shortcut(24, 32)
        self.forward1_2 = mainBlock(32, 32, 3, 1)
        #self.shortcut1_2 = shortcut(32, 32)    
        
        self.forward2_1 = mainBlock(32, 64, 3, 1)
        self.shortcut2_1 = shortcut(32, 64)
        self.forward2_2 = mainBlock(64, 64, 3, 1)
        #self.shortcut2_2 = shortcut(64, 64)
        
        self.forward3_1 = mainBlock(64, 64, 3, 1)
        #self.shortcut3_1 = shortcut(64, 64)
        self.forward3_2 = mainBlock(64, 64, 3, 1)
        #self.shortcut3_2 = shortcut(64, 64)
        self.forward3_3 = mainBlock(64, 64, 3, 1)
        #self.shortcut3_3 = shortcut(64, 64)
        
        self.forward4_1 = mainBlock(64, 64, 3, 1)
        #self.shortcut4_1 = shortcut(64, 64)
        self.forward4_2 = mainBlock(64, 64, 3, 1)
        #self.shortcut4_2 = shortcut(64, 64)
        self.forward4_3 = mainBlock(64, 64, 3, 1)
        #self.shortcut4_3 = shortcut(64, 64)
        
        self.forward4_1 = mainBlock(64, 64, 3, 1)
        #self.shortcut4_1 = shortcut(64, 64)
        self.forward4_2 = mainBlock(64, 64, 3, 1)
        #self.shortcut4_2 = shortcut(64, 64)
        self.forward4_3 = mainBlock(64, 64, 3, 1)
        #self.shortcut4_3 = shortcut(64, 64)
        
        self.back1_1 = nn.Sequential(
                        nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
                        nn.BatchNorm3d(64),
                        nn.ReLU(inplace))
        
        self.backward1_1 = mainBlock(128, 64, 3, 1)
        self.shortcut1_1 = shortcut(128, 64)
        self.backward1_2 = mainBlock(64, 64, 3, 1)
        #self.shortcut1_2 = shortcut(64, 64)
        self.backward1_3 = mainBlock(64, 64, 3, 1)
        #self.shortcut1_3 = shortcut(64, 64)
        
        self.back2_2 = nn.Sequential(
                        nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
                        nn.BatchNorm3d(64),
                        nn.ReLU(inplace))
        
        self.backward2_1 = mainBlock(131, 128, 3, 1)
        self.shortcut2_1 = shortcut(131, 128)
        self.backward2_2 = mainBlock(128, 128, 3, 1)
        #self.shortcut2_2 = shortcut(128, 128)
        self.backward2_3 = mainBlock(128, 128, 3, 1)
        #self.shortcut2_3 = shortcut(128, 128)
        
        self.output = nn.Sequential(
                        nn.Conv3d(128, 64, kernel_size = 1), 
                        nn.ReLU(inplace = True),
                        nn.Conv3d(64, 15, kernel_size = 1))
        
        
        
    def forward(self, x):
        
        out1 = self.preBlock(x)
        
        #-----------blob 1-----------#
        out_a = self.forward1_1(out1)
        out_b = self.shortcut1_1(out1)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.forward1_2(out)
        out = out_a + out
        out2 = self.ReLU(out)        
        out, idx = self.maxpool(out2)
        
        #-----------blob 2-----------#
        out_a = self.forward2_1(out)
        out_b = self.shortcut2_1(out)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.forward2_2(out)
        out = out_a + out
        out3 = self.ReLU(out)
        out, idx = self.maxpool(out3)
        
        #-----------blob 3-----------#
        out_a = self.forward3_1(out)
        out_b = self.shortcut3_1(out)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.forward3_2(out)
        out = out_a + out
        out = self.ReLU(out)
        
        out_a = self.forward3_3(out)
        out = out_a + out
        out4 = self.ReLU(out)
        out = self.maxpool(out4)
        
        #-----------blob 4-----------#
        out_a = self.forward4_1(out)
        out_b = self.shortcut4_1(out)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.forward4_2(out)
        out = out_a + out
        out = self.ReLU(out)
        
        out_a = self.forward4_3(out)
        out = out_a + out
        out5 = self.ReLU(out)
        out = self.maxpool(out5)
        
        #-----------deconv1-----------#
        out = self.back1_1(out)
        comb = torch.cat((out, out4), 1)
        
        out_a = self.backward1_1(comb)
        out_b = self.shortcut1_1(comb)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.backward1_2(comb)
        out_b = self.shortcut1_2(comb)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.backward1_3(comb)
        out_b = self.shortcut1_3(comb)
        out = out_a + out_b
        out = self.ReLU(out)
        
        #------------deconv2----------#
        out = self.back2_2(out)
        comb = torch.cat((out, out3), 1)
        
        out_a = self.backward2_1(comb)
        out_b = self.shortcut2_1(comb)
        out = out_a + out_b
        out = self.ReLU(out)
        
        out_a = self.backward2_2(out)
        out = out_a + out
        out = self.ReLU(out)
        
        out_a = self.backward2_3(out)
        out = out_a + out
        out = self.ReLU(out)
        
        
        out = self.drop(out)
        out = self.output(out)
        
        

def mainBlock(in_channels, out_channels, kernel_size, pad):
    mainBlock = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, pad),
                nn.BatchNorm(out_channels),
                nn.ReLU(inplace),
                nn.Conv3d(out_channels, out_channels, kernel_size, pad),
                nn.BatchNorm(out_channels),
                )
    return mainBlock


def shortcut(in_channels, out_channels):
    shortcut = None
    if in_channels != out_channels:
        shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels,kernel_size = 1, pad = 1),
                    nn.BatchNorm(out_channels))
    return shortcut        
    