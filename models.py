import torch
import torch.nn as nn
from einops import rearrange
from utils_models import conv_1x1, conv_nxn, Transformer

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()

        hidden_dim = int(inp)

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        
    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv2 = conv_1x1(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1(dim, channel)
    
    def forward(self, x):
        # Local representations
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        return x


class TransConvNet(nn.Module):
    def __init__(self, image_size, dims, channels, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn(3, channels[0], stride=2)

        self.list_conv = nn.ModuleList([])
        self.list_conv.append(ConvBlock(channels[0], channels[1], 1))
        self.list_conv.append(ConvBlock(channels[1], channels[2], 2))
        self.list_conv.append(ConvBlock(channels[2], channels[3], 2))
       
        
        self.trans_conv_net = nn.ModuleList([])
        self.trans_conv_net.append(TransformerBlock(dims[0], L[0], channels[3], kernel_size, patch_size, int(dims[0]*2)))
        

        self.conv2 = conv_1x1(channels[4], channels[0])
        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc_image = nn.Linear(7056, 512)
            
        self.act = nn.ReLU()

        self.fc_text_features = nn.Linear(768, 512)
        self.finnal_fc = nn.Linear(1024, 6)


    def forward(self, x, y):
        batch = x.shape[0]
        x = self.conv1(x)
        x = self.list_conv[0](x)
        x = self.list_conv[1](x)
        x = self.list_conv[2](x)
        x = self.trans_conv_net[0](x)
        x = self.conv2(x)
        x = self.pool(x).view(batch, -1)


        feature_image = self.act(self.fc_image(x))
        feature_text = self.act(self.fc_text_features(y.float()).view(batch, -1))

        combine_feature = torch.cat((feature_image, feature_text), dim=1)
        x = self.finnal_fc(combine_feature)
        return x


def get_model_net():
    dims = [32]
    channels = [16, 16, 32, 32, 32]
    return TransConvNet((256, 256), dims, channels)