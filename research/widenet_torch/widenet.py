'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-15 13:43:21
'''
import torch
import torch.nn as nn
import torch.nn.functional as f

# PatchEncoder
class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim, dropout_rate):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate

        self.projection = nn.Linear(num_patches, projection_dim)
        self.position_embedding = nn.Embedding(num_patches + 1, projection_dim)
        self.cls_token = torch.zeros((1, 1, projection_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, patch):
        positions = torch.arange(start=0, end=self.num_patches + 1).unsqueeze(0)
        positions = positions.to(patch.device)
        shape_tensor = patch.mean(dim=1).unsqueeze(1)
        cls_token = torch.zeros_like(shape_tensor)
        patch = torch.cat([cls_token, patch], dim=1)
        # encoded = self.projection(patch) + self.position_embedding(positions)
        encoded = patch + self.position_embedding(positions)
        encoded = self.dropout(encoded)
        return encoded

    def get_config(self):
        config = {
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'dropout_rate': self.dropout_rate
        }

        return config

# mlp
def mlp(hidden_units, dropout_rate):
    layers = []
    layers.append(nn.Linear(hidden_units[1], hidden_units[0]))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(hidden_units[0], hidden_units[1]))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)

# Patches

# Switch

# AttenBlock
class AttnBlock(nn.Module):
    def __init__(self, config):
        super(AttnBlock, self).__init__()
        self.attn_norml = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn_layer = nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.transformer["num_heads"],
                    dropout=config.transformer["attention_dropout_rate"]
                )
        self.attn_dropout = nn.Dropout(config.transformer["dropout_rate"])

        transformer_units = [config.transformer["mlp_dim"], config.hidden_size]

        self.ffn_normal = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_layer = mlp(hidden_units=transformer_units, dropout_rate=config.transformer["dropout_rate"])

    def forward(self, x):
            y = x
            x = self.attn_norml(x)
            x,_ = self.attn_layer(x,x,x)
            x = self.attn_dropout(x)
            x =  y + x
        
            y = x
            x = self.ffn_normal(x)
            x = self.ffn_layer(x)
            output = y + x
            return output
# WideNet
class WideNet(nn.Module):
    def __init__(self,config, 
                 img_size=32, switch_deepth=128, 
                 num_classes=10, use_representation=False, 
                 share_att=False, share_ffn=False, group_deepth=128):
        super(WideNet, self).__init__()
        self.config = config
        self.img_size = img_size
        self.switch_deepth = switch_deepth
        self.num_classes = num_classes
        self.use_representation = use_representation
        self.share_att = share_att
        self.share_ffn = share_ffn
        self.group_deepth = group_deepth

        self.patch_size = self.config.patches["size"][0]
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.hidden_size = config.hidden_size
        self.input_channel = 3

        self.patch_conv = nn.Conv2d(self.input_channel, config.hidden_size, 
                                    kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, self.config.hidden_size, 
                                    self.config.transformer["dropout_rate"])

        self.final_layer_normal = nn.LayerNorm(self.config.hidden_size, eps=1e-6)
        self.global_avg_pool = nn.AvgPool1d(kernel_size=self.config.hidden_size, stride=config.hidden_size)
        self.ffn = nn.Linear(in_features=self.num_patches+1,out_features=self.num_classes)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            block = AttnBlock(config)
            self.transformer_blocks.append(block)

    def forward(self, x):
        # (batch,input,w,h)
        x = self.patch_conv(x)
        # (batch,channel,w,h)
        x = x.reshape(x.shape[0],x.shape[2]*x.shape[3],x.shape[1])
        # (batch,patches,channel)
        x = self.patch_encoder(x)
        # (batch,patches+1,channel)

        for block in self.transformer_blocks:  
            x = block(x)

        x = self.final_layer_normal(x)
        # (batch,patches+1,channel)
        x = self.global_avg_pool(x)
        # (batch,patches+1,1)
        x = torch.squeeze(x)
        x = self.ffn(x)
        # (batch,num_classes)
        x = f.tanh(x)
        # (batch,num_classes)
        out = x
        return out
    