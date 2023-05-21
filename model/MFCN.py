import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .MCMunet_parts import  *
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class MCM1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCM1,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,16,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.Branch3x3=nn.Conv2d(16,16,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.Branch5x5=nn.Conv2d(16,16,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)


    def forward(self, x):
        branch1x1=self.Branch1x1(x)

        branch2_1=self.Branch3x3_1(x)
        branch2_2=self.Branch3x3(branch2_1)

        branch3_1=self.Branch5x5_1(x)
        branch3_2=self.Branch5x5(branch3_1)

        branchpool4_1=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool4_2=self.Branchmax1x1(branchpool4_1)

        outputs=[branch1x1,branch2_2,branch3_2,branchpool4_2]
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)

class MCM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCM2,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,64,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.Branch3x3=nn.Conv2d(64,64,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.Branch5x5=nn.Conv2d(64,64,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)


    def forward(self, x):
        branch1x1=self.Branch1x1(x)

        branch2_1=self.Branch3x3_1(x)
        branch2_2=self.Branch3x3(branch2_1)

        branch3_1=self.Branch5x5_1(x)
        branch3_2=self.Branch5x5(branch3_1)

        branchpool4_1=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool4_2=self.Branchmax1x1(branchpool4_1)

        outputs=[branch1x1,branch2_2,branch3_2,branchpool4_2]
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)
class MCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCM3,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,128,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.Branch3x3=nn.Conv2d(128,128,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.Branch5x5=nn.Conv2d(128,128,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, 128, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)


    def forward(self, x):
        branch1x1=self.Branch1x1(x)

        branch2_1=self.Branch3x3_1(x)
        branch2_2=self.Branch3x3(branch2_1)

        branch3_1=self.Branch5x5_1(x)
        branch3_2=self.Branch5x5(branch3_1)

        branchpool4_1=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool4_2=self.Branchmax1x1(branchpool4_1)

        outputs=[branch1x1,branch2_2,branch3_2,branchpool4_2]
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)
class MCM4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCM4,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,128,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.Branch3x3=nn.Conv2d(128,128,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.Branch5x5=nn.Conv2d(128,128,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, 128, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)


    def forward(self, x):
        branch1x1=self.Branch1x1(x)

        branch2_1=self.Branch3x3_1(x)
        branch2_2=self.Branch3x3(branch2_1)

        branch3_1=self.Branch5x5_1(x)
        branch3_2=self.Branch5x5(branch3_1)

        branchpool4_1=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool4_2=self.Branchmax1x1(branchpool4_1)

        outputs=[branch1x1,branch2_2,branch3_2,branchpool4_2]
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(

            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)


        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConvModule(nn.Module):
    def __init__(self,in_channels, out_channels,):
        super(ConvModule,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=5,dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
class MFCN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MFCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(6, 64)
        self.mcm = MCM1(64, 64)
        # self.conmodel=ConvModule(64,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.mcm2 = MCM2(256, 256)
        self.down3 = Down(256, 512)
        self.mcm3=MCM3(512,512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x1,x2):
        x = torch.cat((x1, x2), 1)
        x1 = self.inc(x)
        x1 = self.mcm(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.mcm2(x3)
        x4 = self.down3(x3)
        x4=self.mcm3(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
if __name__ == "__main__":
    import torch as t
    rgb = t.randn(1, 3,16,16)
    rgb1 = t.randn(1, 3, 16, 16)
    net = MFCN(6,2)
    out = net(rgb,rgb1)
    print(out.shape)