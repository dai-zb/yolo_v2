import torchvision.models as models
import torch.nn as nn

class detnet_bottleneck(nn.Module):
    # type B use 1x1 conv

    def __init__(self, in_planes, planes, use_1x1_conv=False):
        super(detnet_bottleneck, self).__init__()
        # 1x1卷积  调整通道数
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=1, stride=1, bias=False)
        # BN
        self.bn1 = nn.BatchNorm2d(planes)
        
        # LeakyReLU
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        
        # conv
        # 注意，这里参数为 dilation=2，使用了空洞卷积
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=1, padding=2,
                               bias=False, dilation=2)
        
        # BN
        self.bn2 = nn.BatchNorm2d(planes)
        
        # LeakyReLU
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        
        # 1x1卷积  调整通道数
        self.conv3 = nn.Conv2d(planes, planes, 
                               kernel_size=1, stride=1, bias=False)
        
        # BN
        self.bn3 = nn.BatchNorm2d(planes)
        
        # LeakyReLU
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)

        # 用于形成skip layer
        self.downsample = nn.Sequential()  # 空的相当于 *1 操作
        if in_planes != planes or use_1x1_conv:
            self.downsample = nn.Sequential(
                # 使用1x1的卷积，改变通道数量
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=1, bias=False),
                # BN
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        
        out = self.relu2(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))
        
        # downsample 形成了skip layer
        out += self.downsample(x)
        
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    """
    对resnet50进行了包装，但是没有用最后的全局平均池化和全连接
    """
    def __init__(self, resnet_model, c_num, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

        # 复用ResNet的参数
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        self.layer1 = resnet_model.layer1
        # 输出的shape -1,  64/ 256, 112, 112
        self.layer2 = resnet_model.layer2
        # 输出的shape -1, 128/ 512, 56, 56    # 通道*2  下采样  8 
        self.layer3 = resnet_model.layer3
        # 输出的shape -1, 256/1024, 28, 28    # 通道*2  下采样 16
        self.layer4 = resnet_model.layer4
        # 输出的shape -1, 512/2048, 14, 14    # 通道*2  下采样 32
        
        # 新添加的网络
        self.layer5 = nn.Sequential(
            detnet_bottleneck(in_planes=c_num[0], planes=c_num[1]),  # 降低通道，宽高不变
            detnet_bottleneck(in_planes=c_num[1], planes=c_num[1]),   # 通道、宽、高不变
            detnet_bottleneck(in_planes=c_num[1], planes=c_num[1])    # 通道、宽、高不变
        )

        self.conv_end = nn.Conv2d(c_num[1], c_num[2], kernel_size=3,
                                  stride=1, padding=1, bias=False) # 通道、宽、高不变
        # BN
        self.bn_end = nn.BatchNorm2d(c_num[2])

    def forward(self, x):
        # resnet的原本的网络 ===================
        
        # 输入的shape -1, 3, 448, 448
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 输出的shape -1, 2048, 14, 14

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        
        # 新加的层 ============================
        x = self.layer5(x)
        # 输出的shape  -1, 256, 14, 14

        x = self.conv_end(x)
        # 输出的shape  -1, 30, 14, 14

        x = self.bn_end(x)
        
        # x = torch.sigmoid(x)  # 归一化到0-1
        
        # 交换，将C移到后面
        x = x.permute(0, 2, 3, 1)  
        # 输出的shape  -1, 14, 14, 30
        
        return x


def resnet_50_101_152(resnet, c_num=[2048, 256], out_channel=30):
    """
    输入的tensor shape  -1,  3, 32*S, 32*S
    输出的tensor shape  -1,  S,  S,  30
    """
    del resnet.avgpool
    del resnet.fc
    c_num.append(out_channel)
    return ResNet(resnet, c_num)


def resnet_18_34(resnet, c_num=[512, 256], out_channel=30):
    """
    输入的tensor shape  -1,  3, 32*S, 32*S
    输出的tensor shape  -1,  S,  S,  30
    """
    del resnet.avgpool
    del resnet.fc
    c_num.append(out_channel)
    return ResNet(resnet, c_num)

    
# def resnet50(pretrained: bool = False):
#     """
#     输入的tensor shape  -1,  3, 32*S, 32*S
    
#     输出的tensor shape  -1, S,  S,  30
#     """
#     resnet50 = models.resnet50(pretrained=pretrained)
#     del resnet50.avgpool
#     del resnet50.fc
    
#     c_num = [2048, 256, 30]
    
#     return ResNet(resnet50, c_num)


# def resnet34(pretrained: bool = False):
#     """
#     输入的tensor shape  -1,  3, 32*S, 32*S
    
#     输出的tensor shape  -1, S,  S,  30
#     """
#     resnet34 = models.resnet34(pretrained=pretrained)
#     del resnet34.avgpool
#     del resnet34.fc
    
#     c_num = [512, 256, 30]
    
#     return ResNet(resnet34, c_num)


# def resnet18(pretrained: bool = False):
#     """
#     输入的tensor shape  -1,  3, 32*S, 32*S
    
#     输出的tensor shape  -1, S,  S,  30
#     """
#     resnet18 = models.resnet18(pretrained=pretrained)
#     del resnet18.avgpool
#     del resnet18.fc
    
#     c_num = [512, 256, 30]
    
#     return ResNet(resnet18, c_num)


def resnet(name, out_channel: int = 30, pretrained: bool = False):
    if name == 'resnet101':
        return resnet_50_101_152(models.resnet101(pretrained=pretrained),
                                out_channel=out_channel)
    if name == 'resnet50':
        return resnet_50_101_152(models.resnet50(pretrained=pretrained),
                                out_channel=out_channel)
    if name == 'resnet34':
        return resnet_18_34(models.resnet34(pretrained=pretrained),
                                out_channel=out_channel)
    if name == 'resnet18':
        return resnet_18_34(models.resnet18(pretrained=pretrained),
                                out_channel=out_channel)
