import torch.nn as nn

####################################################################
# ResNet Block
####################################################################

'''
Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
'''
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        
        self.relu = nn.ReLU()

    def forward(self, x):
        previous_x = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        previous_x = self.conv_shortcut(previous_x)

        out += previous_x
        out = self.relu(out)

        return out