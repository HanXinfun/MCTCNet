from torch.nn import LayerNorm, Linear, Dropout, Softmax
from einops import rearrange, repeat
import copy
from timm.models.layers import DropPath, trunc_normal_
import re
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torch.nn as nn
import torch
import torch.nn.functional as F
from .SimilarLoss import SimilarLossCalculator


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim , bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
#         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
#         self.attn = Attention(dim = 64)
        self.attn = MCrossAttention(dim = dim)
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x= self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

    

class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads= 8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))
       
    def forward(self, x):
        for layer_block in self.layer:
            x= layer_block(x)
            
        encoded = self.encoder_norm(x)
        return encoded[:,0]


# MCTCNet
class MCTCNet(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, channels, calculator, test):
        super(MCTCNet, self).__init__()

        self.channels = channels
        self.num_channels = len(channels)

        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_channels, FM))

        self.test = test # SimilarLoss
        self.calculator = calculator

        self.features1 = nn.Sequential(
            # [16, 3, 128, 128]
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # [16, 64, 31, 31]
            nn.ReLU(inplace=True),  # [16, 64, 31, 31]
            nn.MaxPool2d(kernel_size=3, stride=2),  # torch.Size([16, 64, 15, 15])
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # [16, 192, 15, 15]
            nn.ReLU(inplace=True),  # [16, 192, 15, 15]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [16, 192, 7, 7]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # [16, 384, 7, 7]
            nn.ReLU(inplace=True),  # [16, 384, 7, 7]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # [16, 256, 7, 7]
            nn.ReLU(inplace=True),  # [16, 256, 7, 7]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # [16, 256, 7, 7]
            nn.ReLU(inplace=True),  # [16, 256, 7, 7]
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # [16, 128, 7, 7]
            nn.ReLU(inplace=True),  # [16, 128, 7, 7]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [16, 128, 3, 3]
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.trans = TransformerEncoder(FM)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(FM , Classes)


    def token_x(self,x):
        x = x.reshape(x.shape[0], -1) # [2, 1152]
        x = x.unsqueeze(1) # [2, 1, 1152]
        return x


    def final_layers(self, x):
        # embeddings
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        # TransformerEncoder
        x = self.trans(embeddings)  # [2, 128]
        x = x.reshape(x.shape[0], -1)
        out = self.out(x)  # [2, 6]
        return out


    def forward(self, x):
        x_all = []
        num_channels = x.shape[1]
        channels = self.channels

        if 'rgb' in channels:
            x_rgb = x[:, :3, :, :]
            x_rgb = self.features1(x_rgb)
            x_all.append(x_rgb)
            rgb_index = channels.index('rgb')
            other_channels = channels[rgb_index + 1:]
        else:
            other_channels = channels
            rgb_index = -1  # no rgb

        # else(not include rgb)
        for idx, channel in enumerate(other_channels):
            actual_idx = idx + (3 if rgb_index != -1 else 0) 
            if actual_idx < x.shape[1]:
                x_channel = x[:, actual_idx:actual_idx + 1, :, :]

                # data clean
                if x_channel.min() < -32768:
                    x_channel = torch.clamp(x_channel, min=-32768, max=32768)

                x_channel = self.features2(x_channel)
                x_all.append(x_channel)



        x_processed = [self.token_x(x_i) for x_i in x_all] # [2, 1, 1152]
        x_concatenated = torch.cat(x_processed, dim=1)

        # transformer
        x_out = self.final_layers(x_concatenated)

        if len(channels) > 1 and self.test == 2:
            # If SimilarLoss
            similar_loss = self.calculator.calculate_similar_loss(x_processed)
            return x_out, similar_loss
        else:
            return x_out
        