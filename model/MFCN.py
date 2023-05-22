import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .MCMunet_parts import  *
class OneConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
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
class ThreeConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FourConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.double_conv(x)
class MCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCM,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,out_channels//4,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.Branch3x3=nn.Conv2d(out_channels//4,out_channels//4,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.Branch5x5=nn.Conv2d(out_channels//4,out_channels//4,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)

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

    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(

            # DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


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
        self.d1=DoubleConv(6,64)
        self.mcm1=MCM(64,64)
        self.down1=Down()
        self.d2 = DoubleConv(64, 128)
        self.down2 = Down()
        self.t1=ThreeConv(128,256)
        self.mcm2 = MCM(256,256)
        self.down3 = Down()
        self.t2 = ThreeConv(256, 512)
        self.mcm3 = MCM(512, 512)
        self.down4= Down()

        self.up1=Up()
        self.f1=FourConv(1024,512)
        self.up2 = Up()
        self.t3=ThreeConv(768,256)
        self.up3 = Up()
        self.d3 = ThreeConv(384, 128)
        self.up4=Up()
        self.one=OneConv(192,64)
        self.outc = OutConv(64, n_classes)
    def forward(self, x1,x2):
        x = torch.cat((x1, x2), 1)
        x1=self.d1(x)
        x1=self.mcm1(x1)
        x2=self.down1(x1)
        x2=self.d2(x2)
        x3=self.down2(x2)
        x3=self.t1(x3)
        x3=self.mcm2(x3)
        x4=self.down3(x3)
        x4=self.t2(x4)
        x4=self.mcm3(x4)
        x5=self.down4(x4)


        x5=self.up1(x5,x4)
        x5=self.f1(x5)
        x5=self.up2(x5,x3)
        x5=self.t3(x5)
        x5=self.up3(x5,x2)
        x5=self.d3(x5)
        x5=self.up4(x5,x1)
        x5=self.one(x5)
        logits = self.outc(x5)
        return logits
if __name__ == "__main__":
    import torch as t
    rgb = t.randn(1, 3,16,16)
    rgb1 = t.randn(1, 3, 16, 16)
    net = MFCN(6,2)
    out = net(rgb,rgb1)
    print(out.shape)