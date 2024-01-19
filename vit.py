import torch 
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels,
                                    emb_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        

    def forward(self, x):
        x = self.projection(x)  
        x = x.flatten(2)      
        x = x.transpose(1, 2)   
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=12, n_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.TransformerEncoderLayer(d_model=emb_size,
                                                          nhead=n_heads,
                                                          dim_feedforward=int(emb_size * mlp_ratio),
                                                          dropout=0.1))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self,  image_size=224, patch_size=16, num_classes=196):
        super().__init__()
        h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        emb_size = patch_size * patch_size * 3  
        
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_size=emb_size)
        num_patches = (h // patch_size) * (w // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.transformer_encoder = TransformerEncoder(emb_size=emb_size)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        x = self.transformer_encoder(x)
        x = self.mlp_head(x[:, 0])
        return x    
    
 