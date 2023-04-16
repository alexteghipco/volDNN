import torch.nn as nn
import torch

class vgg2(nn.Module):
    """
    VGG version 2.
    """

    def __init__(self, drop, model_depth, num_classes, in_channels):
        super().__init__()

        # Define convolutional block configurations based on number of layers
        if model_depth == 1:
            cfg = [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M']
        elif model_depth == 2:
            #cfg = [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M']
            cfg = [8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M']
        elif model_depth == 3:
            cfg = [8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M']
        elif model_depth == 4:
            cfg = [8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128]
        else:
            raise ValueError('Invalid number of layers')

        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]#, padding=0, dilation=1, ceil_mode=False)]
            else:
                if not layers:
                    layers += [nn.Conv3d(in_channels, v, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv3d(v, v, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                    in_channels = v
                else:
                    layers += [nn.Conv3d(in_channels, v, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                    in_channels = v


        self.features = nn.Sequential(*layers)
        #self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels*2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_channels*2, affine=True),
            nn.Dropout(p=drop),
            nn.Linear(in_channels*2, in_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_classes, bias=True)
            #nn.Softmax(dim=1)
            #nn.LogSoftmax(dim=1)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    #def forward(self, x):
    ## Handle smaller input tensor
    #    if x.size()[2] < 8 or x.size()[3] < 8 or x.size()[4] < 8:
    #        x = nn.functional.pad(x, (0,0,0,0,8-x.size()[2],8-x.size()[3],8-x.size()[4]))
    #    x = self.features(x)
    #    #x = self.avg
    #    x = x.view(x.size(0), -1)
    #    x = self.classifier(x)
    #    return x


def generate_model(drop, model_depth, num_classes, in_channels):
    assert model_depth in [1,2,3,4]

    model = vgg2(drop, model_depth, num_classes, in_channels)

    return model

